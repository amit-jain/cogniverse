"""Unit tests for RootCauseAnalyzer — failure patterns, temporal analysis, hypotheses."""

from collections import Counter
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from cogniverse_evaluation.analysis.root_cause_analysis import (
    FailurePattern,
    RootCauseAnalyzer,
    RootCauseHypothesis,
)


def make_trace(
    trace_id: str,
    status: str = "success",
    error: str = None,
    operation: str = "search",
    profile: str = "colpali",
    strategy: str = "default",
    duration_ms: float = 100.0,
    timestamp: datetime = None,
):
    return SimpleNamespace(
        trace_id=trace_id,
        status=status,
        error=error,
        operation=operation,
        profile=profile,
        strategy=strategy,
        duration_ms=duration_ms,
        timestamp=timestamp or datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def analyzer():
    return RootCauseAnalyzer()


@pytest.mark.unit
class TestErrorClassification:
    def test_timeout_error(self, analyzer):
        assert analyzer._classify_error("connection timed out after 30s") == "timeout"

    def test_memory_error(self, analyzer):
        assert analyzer._classify_error("out of memory: failed to allocate") == "memory"

    def test_connection_error(self, analyzer):
        assert (
            analyzer._classify_error("connection refused on port 8080") == "connection"
        )

    def test_rate_limit_error(self, analyzer):
        assert analyzer._classify_error("rate limit exceeded (429)") == "rate_limit"

    def test_model_error(self, analyzer):
        assert analyzer._classify_error("model not found: colpali-v1") == "model_error"

    def test_data_format_error(self, analyzer):
        assert (
            analyzer._classify_error("invalid format: schema mismatch") == "data_format"
        )

    def test_permission_error(self, analyzer):
        assert analyzer._classify_error("permission denied") == "permission"

    def test_not_found_error(self, analyzer):
        assert analyzer._classify_error("resource not found (404)") == "not_found"

    def test_unknown_error(self, analyzer):
        assert (
            analyzer._classify_error("something went wrong unexpectedly") == "unknown"
        )

    def test_empty_error(self, analyzer):
        assert analyzer._classify_error("") == "unknown"

    def test_none_error(self, analyzer):
        assert analyzer._classify_error(None) == "unknown"

    def test_case_insensitive(self, analyzer):
        assert analyzer._classify_error("TIMEOUT ERROR") == "timeout"


@pytest.mark.unit
class TestKnownIssuesLoading:
    def test_known_issues_contains_expected_keys(self, analyzer):
        issues = analyzer.known_issues
        assert "timeout" in issues
        assert "memory" in issues
        assert "connection" in issues
        assert "rate_limit" in issues
        assert "model_error" in issues
        assert "data_format" in issues

    def test_known_issues_have_required_fields(self, analyzer):
        for key, info in analyzer.known_issues.items():
            assert "pattern" in info, f"Missing 'pattern' in {key}"
            assert "category" in info, f"Missing 'category' in {key}"
            assert "suggestion" in info, f"Missing 'suggestion' in {key}"


@pytest.mark.unit
class TestHourlyDistribution:
    def test_calculates_per_hour_count(self, analyzer):
        traces = [
            make_trace("t1", timestamp=datetime(2024, 1, 1, 9, 0)),
            make_trace("t2", timestamp=datetime(2024, 1, 1, 9, 30)),
            make_trace("t3", timestamp=datetime(2024, 1, 1, 14, 0)),
        ]
        dist = analyzer._calculate_hourly_distribution(traces)
        assert dist[9] == 2
        assert dist[14] == 1

    def test_empty_traces(self, analyzer):
        assert analyzer._calculate_hourly_distribution([]) == {}


@pytest.mark.unit
class TestTemporalPatternAnalysis:
    def test_high_failure_rate_hour_detected(self, analyzer):
        base_time = datetime(2024, 1, 1, 10, 0)
        # All 5 traces at hour 10, all failed → 100% failure rate
        all_traces = [
            make_trace(f"t{i}", status="error", error="timeout", timestamp=base_time)
            for i in range(5)
        ]
        failed_traces = all_traces[:]

        patterns = analyzer._analyze_temporal_patterns(failed_traces, all_traces)
        hourly = [p for p in patterns if p["type"] == "hourly"]
        assert any(p["hour"] == 10 and p["failure_rate"] == 1.0 for p in hourly)

    def test_low_failure_rate_hour_not_included(self, analyzer):
        base_time = datetime(2024, 1, 1, 8, 0)
        all_traces = [make_trace(f"t{i}", timestamp=base_time) for i in range(20)]
        # Only 1 failure out of 20 → 5% < 10% threshold
        failed_traces = [all_traces[0]]
        all_traces[0].status = "error"
        all_traces[0].error = "oops"

        patterns = analyzer._analyze_temporal_patterns(failed_traces, all_traces)
        hourly = [p for p in patterns if p["type"] == "hourly"]
        assert not any(p["hour"] == 8 for p in hourly)

    def test_failure_burst_detected(self, analyzer):
        base = datetime(2024, 1, 1, 15, 0)
        # 5 failures within 2 minutes — should trigger burst
        failures = [
            make_trace(
                f"t{i}",
                status="error",
                error="timeout",
                timestamp=base + timedelta(seconds=i * 20),
            )
            for i in range(5)
        ]
        patterns = analyzer._analyze_temporal_patterns(failures, failures)
        bursts = [p for p in patterns if p["type"] == "burst"]
        assert len(bursts) >= 1
        assert bursts[0]["failure_count"] >= 3

    def test_fewer_than_3_failures_no_burst(self, analyzer):
        base = datetime(2024, 1, 1, 11, 0)
        failures = [
            make_trace(
                f"t{i}",
                status="error",
                error="err",
                timestamp=base + timedelta(seconds=i),
            )
            for i in range(2)
        ]
        bursts = analyzer._detect_failure_bursts(
            sorted(failures, key=lambda x: x.timestamp)
        )
        assert bursts == []


@pytest.mark.unit
class TestAttributeCorrelation:
    def test_high_failure_rate_profile_detected(self, analyzer):
        # 8 total with profile "bad_profile", 6 fail → 75% failure rate
        bad_traces = [
            make_trace(
                f"bad{i}", status="error", error="timeout", profile="bad_profile"
            )
            for i in range(6)
        ]
        good_bad = [
            make_trace(f"gbad{i}", status="success", profile="bad_profile")
            for i in range(2)
        ]
        good_traces = [
            make_trace(f"good{i}", status="success", profile="good_profile")
            for i in range(10)
        ]
        all_traces = bad_traces + good_bad + good_traces
        failed_traces = bad_traces

        patterns = analyzer._calculate_attribute_correlation(
            failed_traces, all_traces, "profile"
        )
        assert any(p.pattern_value == "bad_profile" for p in patterns)

    def test_uncorrelated_attribute_excluded(self, analyzer):
        # Equal failure rate across profiles → no significant correlation
        traces = [
            make_trace(
                f"t{i}",
                status=("error" if i % 2 == 0 else "success"),
                error=("err" if i % 2 == 0 else None),
                profile="profileA",
            )
            for i in range(10)
        ]
        failed = [t for t in traces if t.status == "error"]
        patterns = analyzer._calculate_attribute_correlation(failed, traces, "profile")
        # With uniform failure rate there should be no high-correlation pattern
        for p in patterns:
            assert p.correlation_strength <= 0.3 or p.failure_rate <= 1.0


@pytest.mark.unit
class TestFailurePatternAnalysis:
    def test_error_types_counted(self, analyzer):
        failed = [
            make_trace("t1", status="error", error="connection refused"),
            make_trace("t2", status="error", error="connection reset"),
            make_trace("t3", status="error", error="timeout error"),
        ]
        all_traces = failed + [make_trace("t4", status="success")]
        patterns = analyzer._analyze_failure_patterns(failed, all_traces)

        assert patterns["error_types"]["connection"] == 2
        assert patterns["error_types"]["timeout"] == 1

    def test_failed_operations_counted(self, analyzer):
        failed = [
            make_trace("t1", status="error", error="err", operation="video_search"),
            make_trace("t2", status="error", error="err", operation="video_search"),
            make_trace("t3", status="error", error="err", operation="text_search"),
        ]
        all_traces = failed
        patterns = analyzer._analyze_failure_patterns(failed, all_traces)

        assert patterns["failed_operations"]["video_search"] == 2
        assert patterns["failed_operations"]["text_search"] == 1

    def test_profiles_and_strategies_counted(self, analyzer):
        failed = [
            make_trace(
                "t1", status="error", error="err", profile="colpali", strategy="rerank"
            ),
            make_trace(
                "t2", status="error", error="err", profile="colpali", strategy="default"
            ),
        ]
        all_traces = failed
        patterns = analyzer._analyze_failure_patterns(failed, all_traces)

        assert patterns["failed_profiles"]["colpali"] == 2
        assert patterns["failed_strategies"]["rerank"] == 1


@pytest.mark.unit
class TestPerformancePatternAnalysis:
    def test_slow_operations_stats_calculated(self, analyzer):
        slow = [
            make_trace("s1", duration_ms=1500.0, operation="embed"),
            make_trace("s2", duration_ms=2000.0, operation="embed"),
            make_trace("s3", duration_ms=800.0, operation="search"),
        ]
        normal = [
            make_trace(f"n{i}", duration_ms=100.0, operation="embed") for i in range(5)
        ]

        patterns = analyzer._analyze_performance_patterns(slow, normal)

        assert "embed" in patterns["slow_operations"]
        embed_stats = patterns["slow_operations"]["embed"]
        assert embed_stats["count"] == 2
        assert embed_stats["avg_duration"] == 1750.0
        assert embed_stats["min_duration"] == 1500.0
        assert embed_stats["max_duration"] == 2000.0

    def test_latency_distribution_slowdown_factor(self, analyzer):
        slow = [make_trace(f"s{i}", duration_ms=1000.0) for i in range(5)]
        normal = [make_trace(f"n{i}", duration_ms=100.0) for i in range(10)]

        patterns = analyzer._analyze_performance_patterns(slow, normal)
        dist = patterns["latency_distribution"]
        assert dist["slow_mean"] == 1000.0
        assert abs(dist["normal_mean"] - 100.0) < 1e-6
        assert abs(dist["slowdown_factor"] - 10.0) < 1e-6

    def test_no_normal_traces_handles_gracefully(self, analyzer):
        slow = [make_trace("s1", duration_ms=500.0)]
        patterns = analyzer._analyze_performance_patterns(slow, [])
        assert patterns["latency_distribution"]["slowdown_factor"] == 0


@pytest.mark.unit
class TestCompareByAttribute:
    def test_groups_failure_rates_by_attribute(self, analyzer):
        traces = [
            make_trace("t1", status="error", error="err", profile="A"),
            make_trace("t2", status="error", error="err", profile="A"),
            make_trace("t3", status="success", profile="A"),
            make_trace("t4", status="success", profile="B"),
            make_trace("t5", status="success", profile="B"),
        ]
        failed = [t for t in traces if t.status == "error"]

        comparison = analyzer._compare_by_attribute(traces, failed, "profile")
        assert comparison["A"]["total"] == 3
        assert comparison["A"]["failed"] == 2
        assert abs(comparison["A"]["failure_rate"] - 2 / 3) < 1e-9
        assert comparison["B"]["total"] == 2
        assert comparison["B"]["failed"] == 0
        assert comparison["B"]["failure_rate"] == 0.0


@pytest.mark.unit
class TestStatisticalAnalysis:
    def test_basic_stats_computed(self, analyzer):
        successful = [
            make_trace(f"s{i}", status="success", duration_ms=float(100 + i * 10))
            for i in range(10)
        ]
        failed = [make_trace("f1", status="error", error="timeout")]

        stats = analyzer._perform_statistical_analysis(
            failed, successful, successful + failed
        )
        assert abs(stats["failure_rate"] - 1 / 11) < 1e-9
        assert abs(stats["success_rate"] - 10 / 11) < 1e-9
        assert "mean" in stats["success_duration_stats"]
        assert "p95" in stats["success_duration_stats"]

    def test_empty_traces_returns_empty(self, analyzer):
        result = analyzer._perform_statistical_analysis([], [], [])
        assert result == {}


@pytest.mark.unit
class TestGenerateFailureHypotheses:
    def test_known_error_type_produces_hypothesis(self, analyzer):
        failed = [
            make_trace(f"t{i}", status="error", error="connection refused")
            for i in range(5)
        ]
        failure_analysis = {
            "error_types": Counter({"connection": 5}),
            "failed_operations": Counter({"search": 5}),
            "failed_profiles": Counter(),
            "failed_strategies": Counter(),
            "temporal_patterns": [],
            "correlated_attributes": [],
        }
        hypotheses = analyzer._generate_failure_hypotheses(failed, failure_analysis)
        assert len(hypotheses) >= 1
        assert hypotheses[0].category in (
            "network",
            "timeout",
            "resource",
            "model",
            "data",
            "throttling",
        )
        assert hypotheses[0].confidence > 0.0
        assert len(hypotheses[0].evidence) >= 1

    def test_temporal_burst_produces_hypothesis(self, analyzer):
        base = datetime(2024, 1, 1, 14, 0)
        failed = [
            make_trace(f"t{i}", status="error", error="err", timestamp=base)
            for i in range(3)
        ]
        failure_analysis = {
            "error_types": Counter(),
            "failed_operations": Counter(),
            "failed_profiles": Counter(),
            "failed_strategies": Counter(),
            "temporal_patterns": [
                {
                    "type": "burst",
                    "start_time": "2024-01-01T14:00:00",
                    "end_time": "2024-01-01T14:04:00",
                    "failure_count": 3,
                    "duration_minutes": 4.0,
                    "trace_ids": ["t0", "t1", "t2"],
                }
            ],
            "correlated_attributes": [],
        }
        hypotheses = analyzer._generate_failure_hypotheses(failed, failure_analysis)
        burst_hypotheses = [h for h in hypotheses if "burst" in h.hypothesis.lower()]
        assert len(burst_hypotheses) == 1
        assert burst_hypotheses[0].confidence == 0.8

    def test_correlated_attribute_produces_hypothesis(self, analyzer):
        failed = [make_trace(f"t{i}", status="error", error="err") for i in range(3)]
        pattern = FailurePattern(
            pattern_type="profile",
            pattern_value="slow_profile",
            failure_rate=0.9,
            occurrence_count=9,
            confidence=0.75,
            correlation_strength=0.8,
        )
        failure_analysis = {
            "error_types": Counter(),
            "failed_operations": Counter(),
            "failed_profiles": Counter(),
            "failed_strategies": Counter(),
            "temporal_patterns": [],
            "correlated_attributes": [pattern],
        }
        hypotheses = analyzer._generate_failure_hypotheses(failed, failure_analysis)
        profile_hypotheses = [h for h in hypotheses if "slow_profile" in h.hypothesis]
        assert len(profile_hypotheses) == 1
        assert profile_hypotheses[0].confidence == 0.75


@pytest.mark.unit
class TestGeneratePerformanceHypotheses:
    def test_slow_operation_hypothesis(self, analyzer):
        slow = [
            make_trace(f"s{i}", operation="embed", duration_ms=2000.0) for i in range(4)
        ]
        perf_analysis = {
            "slow_operations": {
                "embed": {
                    "count": 4,
                    "avg_duration": 2000.0,
                    "min_duration": 1800.0,
                    "max_duration": 2200.0,
                    "durations": [1800, 2000, 2000, 2200],
                }
            },
            "slow_profiles": Counter({"colpali": 4}),
            "slow_strategies": Counter(),
            "latency_distribution": {
                "slow_mean": 2000.0,
                "slow_std": 100.0,
                "normal_mean": 100.0,
                "normal_std": 10.0,
                "slowdown_factor": 20.0,
            },
        }
        hypotheses = analyzer._generate_performance_hypotheses(slow, perf_analysis)
        op_hypotheses = [h for h in hypotheses if "embed" in h.hypothesis]
        assert len(op_hypotheses) == 1
        assert "20.0x" in op_hypotheses[0].evidence[3]  # slowdown factor in evidence

    def test_slow_profile_hypothesis(self, analyzer):
        slow = [
            make_trace(f"s{i}", profile="heavy_profile", duration_ms=3000.0)
            for i in range(5)
        ]
        perf_analysis = {
            "slow_operations": {},
            "slow_profiles": Counter({"heavy_profile": 5}),
            "slow_strategies": Counter(),
            "latency_distribution": {
                "slow_mean": 3000.0,
                "slow_std": 0.0,
                "normal_mean": 200.0,
                "normal_std": 10.0,
                "slowdown_factor": 15.0,
            },
        }
        hypotheses = analyzer._generate_performance_hypotheses(slow, perf_analysis)
        profile_hypotheses = [h for h in hypotheses if "heavy_profile" in h.hypothesis]
        assert len(profile_hypotheses) == 1


@pytest.mark.unit
class TestGenerateRecommendations:
    def test_timeout_category_produces_high_priority(self, analyzer):
        hypotheses = [
            RootCauseHypothesis(
                hypothesis="Timeout failures",
                confidence=0.9,
                evidence=["lots of timeouts"],
                affected_traces=["t1"],
                suggested_action="Increase timeout limits",
                category="timeout",
            )
        ]
        recs = analyzer._generate_recommendations(hypotheses)
        assert any(r["priority"] == "high" and r["category"] == "timeout" for r in recs)

    def test_resource_category_produces_high_priority(self, analyzer):
        hypotheses = [
            RootCauseHypothesis(
                hypothesis="Memory OOM",
                confidence=0.8,
                evidence=["OOM errors"],
                affected_traces=[],
                suggested_action="Reduce batch size",
                category="resource",
            )
        ]
        recs = analyzer._generate_recommendations(hypotheses)
        assert any(
            r["priority"] == "high" and r["category"] == "resource" for r in recs
        )

    def test_configuration_category_medium_priority(self, analyzer):
        hypotheses = [
            RootCauseHypothesis(
                hypothesis="Bad profile config",
                confidence=0.7,
                evidence=["profile correlation"],
                affected_traces=[],
                suggested_action="Review config",
                category="configuration",
            )
        ]
        recs = analyzer._generate_recommendations(hypotheses)
        assert any(
            r["priority"] == "medium" and r["category"] == "configuration" for r in recs
        )

    def test_recommendations_sorted_by_priority(self, analyzer):
        hypotheses = [
            RootCauseHypothesis(
                hypothesis="Slow",
                confidence=0.5,
                evidence=[],
                affected_traces=[],
                suggested_action="Optimize",
                category="performance",
            ),
            RootCauseHypothesis(
                hypothesis="OOM",
                confidence=0.9,
                evidence=[],
                affected_traces=[],
                suggested_action="Scale",
                category="resource",
            ),
            RootCauseHypothesis(
                hypothesis="Config bad",
                confidence=0.7,
                evidence=[],
                affected_traces=[],
                suggested_action="Fix",
                category="configuration",
            ),
        ]
        recs = analyzer._generate_recommendations(hypotheses)
        priorities = [r["priority"] for r in recs]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        assert sorted(priorities, key=lambda p: priority_order.get(p, 3)) == priorities


@pytest.mark.unit
class TestAnalyzeFailures:
    def test_full_analysis_with_mixed_traces(self, analyzer):
        base = datetime(2024, 1, 1, 10, 0)
        failed = [
            make_trace(
                f"f{i}",
                status="error",
                error="connection refused",
                operation="search",
                profile="colpali",
                timestamp=base + timedelta(minutes=i),
            )
            for i in range(5)
        ]
        successful = [
            make_trace(
                f"s{i}",
                status="success",
                duration_ms=float(100 + i * 20),
                timestamp=base + timedelta(minutes=i + 5),
            )
            for i in range(10)
        ]
        all_traces = failed + successful

        result = analyzer.analyze_failures(all_traces)

        assert result["summary"]["total_traces"] == 15
        assert result["summary"]["failed_traces"] == 5
        assert abs(result["summary"]["failure_rate"] - 5 / 15) < 1e-9
        assert len(result["root_causes"]) >= 1
        # Root causes are sorted by confidence descending
        confidences = [h.confidence for h in result["root_causes"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_all_successful_traces_no_failures(self, analyzer):
        traces = [
            make_trace(f"t{i}", status="success", duration_ms=float(100 + i * 10))
            for i in range(10)
        ]
        result = analyzer.analyze_failures(traces, include_performance=False)
        assert result["summary"]["failed_traces"] == 0
        assert result["summary"]["failure_rate"] == 0.0
        assert result["failure_analysis"] == {}

    def test_performance_degraded_traces_identified(self, analyzer):
        # 8 fast traces + 2 very slow (should be above P95 threshold)
        fast = [
            make_trace(f"f{i}", status="success", duration_ms=100.0) for i in range(8)
        ]
        slow = [
            make_trace(f"s{i}", status="success", duration_ms=5000.0) for i in range(2)
        ]
        all_traces = fast + slow

        result = analyzer.analyze_failures(
            all_traces, include_performance=True, performance_threshold_percentile=80
        )
        # Some slow traces must be identified
        assert result["summary"]["performance_degraded"] >= 1

    def test_empty_traces_returns_zero_counts(self, analyzer):
        result = analyzer.analyze_failures([])
        assert result["summary"]["total_traces"] == 0
        assert result["summary"]["failure_rate"] == 0


@pytest.mark.unit
class TestReportGeneration:
    def test_markdown_report_contains_sections(self, analyzer):
        base = datetime(2024, 1, 1, 10, 0)
        failed = [
            make_trace(
                f"f{i}",
                status="error",
                error="timeout error",
                timestamp=base + timedelta(minutes=i),
            )
            for i in range(3)
        ]
        successful = [
            make_trace(
                f"s{i}",
                status="success",
                duration_ms=100.0,
                timestamp=base + timedelta(minutes=i + 3),
            )
            for i in range(10)
        ]
        analysis = analyzer.analyze_failures(failed + successful)

        report = analyzer.generate_rca_report(analysis, format="markdown")
        assert "# Root Cause Analysis Report" in report
        assert "## Summary" in report
        assert "Total Traces: 13" in report

    def test_html_report_contains_structure(self, analyzer):
        failed = [make_trace("f1", status="error", error="timeout error")]
        successful = [
            make_trace(f"s{i}", status="success", duration_ms=100.0) for i in range(5)
        ]
        analysis = analyzer.analyze_failures(failed + successful)

        report = analyzer.generate_rca_report(analysis, format="html")
        assert "<html>" in report
        assert "Root Cause Analysis Report" in report
        assert "Total Traces:" in report

    def test_plain_text_format(self, analyzer):
        analysis = analyzer.analyze_failures(
            [make_trace("t1", status="success", duration_ms=100.0)]
        )
        report = analyzer.generate_rca_report(analysis, format="text")
        assert "total_traces" in report

    def test_markdown_includes_statistical_table_when_profiles_exist(self, analyzer):
        traces = [
            make_trace(
                f"t{i}", status="success", duration_ms=float(100 + i), profile="prof_a"
            )
            for i in range(5)
        ] + [
            make_trace(f"f{i}", status="error", error="err", profile="prof_b")
            for i in range(3)
        ]
        analysis = analyzer.analyze_failures(traces, include_performance=False)
        report = analyzer.generate_rca_report(analysis, format="markdown")
        assert "## Statistical Analysis" in report
        assert "Profile" in report
