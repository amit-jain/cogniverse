"""
Root Cause Analysis for Phoenix Traces

This module provides automated and semi-automated root cause analysis
for failures and performance issues in traces.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy import stats
import re

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a pattern associated with failures"""
    pattern_type: str  # 'operation', 'profile', 'strategy', 'time', 'parameter'
    pattern_value: Any
    failure_rate: float
    occurrence_count: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    correlation_strength: float = 0.0


@dataclass
class RootCauseHypothesis:
    """Hypothesis for root cause"""
    hypothesis: str
    confidence: float
    evidence: List[str]
    affected_traces: List[str]
    suggested_action: str
    category: str  # 'configuration', 'resource', 'timeout', 'data', 'model'
    patterns: List[FailurePattern] = field(default_factory=list)


class RootCauseAnalyzer:
    """Automated root cause analysis for trace failures and performance issues"""
    
    def __init__(self):
        self.failure_patterns = defaultdict(list)
        self.performance_baselines = {}
        self.known_issues = self._load_known_issues()
    
    def _load_known_issues(self) -> Dict[str, Dict]:
        """Load known issue patterns and their solutions"""
        return {
            "timeout": {
                "pattern": r"(timeout|timed out|deadline exceeded)",
                "category": "timeout",
                "suggestion": "Increase timeout limits or optimize query processing"
            },
            "memory": {
                "pattern": r"(out of memory|oom|memory error|allocation failed)",
                "category": "resource",
                "suggestion": "Reduce batch size or increase memory allocation"
            },
            "connection": {
                "pattern": r"(connection refused|connection reset|network error)",
                "category": "network",
                "suggestion": "Check service health and network connectivity"
            },
            "rate_limit": {
                "pattern": r"(rate limit|too many requests|429)",
                "category": "throttling",
                "suggestion": "Implement backoff strategy or increase rate limits"
            },
            "model_error": {
                "pattern": r"(model not found|model error|inference failed)",
                "category": "model",
                "suggestion": "Verify model deployment and configuration"
            },
            "data_format": {
                "pattern": r"(invalid format|parsing error|schema mismatch)",
                "category": "data",
                "suggestion": "Validate input data format and schema"
            }
        }
    
    def analyze_failures(
        self,
        traces: List[Any],
        include_performance: bool = True,
        performance_threshold_percentile: int = 95
    ) -> Dict[str, Any]:
        """
        Perform comprehensive root cause analysis on traces
        
        Args:
            traces: List of trace objects
            include_performance: Include performance degradation analysis
            performance_threshold_percentile: Percentile for performance threshold
            
        Returns:
            Dictionary with root cause analysis results
        """
        # Separate failed and successful traces
        failed_traces = [t for t in traces if t.status == "error" or t.error]
        successful_traces = [t for t in traces if t.status == "success"]
        
        # Performance degraded traces (slow but successful)
        performance_degraded = []
        if include_performance and successful_traces:
            durations = [t.duration_ms for t in successful_traces]
            threshold = np.percentile(durations, performance_threshold_percentile)
            performance_degraded = [
                t for t in successful_traces 
                if t.duration_ms > threshold
            ]
            
            # Log threshold info for debugging
            logger.info(f"RCA Performance Analysis Debug:")
            logger.info(f"  Total traces analyzed: {len(traces)}")
            logger.info(f"  Successful traces: {len(successful_traces)}")
            logger.info(f"  Duration range: {min(durations):.2f}ms - {max(durations):.2f}ms")
            logger.info(f"  Duration percentiles: P50={np.percentile(durations, 50):.2f}ms, P95={np.percentile(durations, 95):.2f}ms, P99={np.percentile(durations, 99):.2f}ms")
            logger.info(f"  Performance threshold (P{performance_threshold_percentile}): {threshold:.2f}ms")
            logger.info(f"  Found {len(performance_degraded)} slow traces")
            
            if performance_degraded:
                sample_slow = performance_degraded[:3]
                for trace in sample_slow:
                    logger.info(f"    - {trace.operation}: {trace.duration_ms:.2f}ms (>{threshold:.2f}ms)")
        
        analysis = {
            "summary": {
                "total_traces": len(traces),
                "failed_traces": len(failed_traces),
                "performance_degraded": len(performance_degraded),
                "failure_rate": len(failed_traces) / len(traces) if traces else 0,
                "analysis_time": datetime.now().isoformat()
            },
            "failure_analysis": {},
            "performance_analysis": {},
            "root_causes": [],
            "recommendations": []
        }
        
        # Analyze failures
        if failed_traces:
            analysis["failure_analysis"] = self._analyze_failure_patterns(
                failed_traces, traces
            )
            failure_hypotheses = self._generate_failure_hypotheses(
                failed_traces, analysis["failure_analysis"]
            )
            analysis["root_causes"].extend(failure_hypotheses)
        
        # Analyze performance degradation
        if performance_degraded:
            analysis["performance_analysis"] = self._analyze_performance_patterns(
                performance_degraded, successful_traces
            )
            # Add threshold info
            analysis["performance_analysis"]["threshold"] = threshold
            analysis["performance_analysis"]["threshold_percentile"] = performance_threshold_percentile
            perf_hypotheses = self._generate_performance_hypotheses(
                performance_degraded, analysis["performance_analysis"]
            )
            analysis["root_causes"].extend(perf_hypotheses)
        
        # Sort hypotheses by confidence
        analysis["root_causes"].sort(key=lambda x: x.confidence, reverse=True)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(
            analysis["root_causes"]
        )
        
        # Add statistical analysis
        analysis["statistical_analysis"] = self._perform_statistical_analysis(
            failed_traces, successful_traces, traces
        )
        
        return analysis
    
    def _analyze_failure_patterns(
        self,
        failed_traces: List[Any],
        all_traces: List[Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in failed traces"""
        patterns = {
            "error_types": Counter(),
            "failed_operations": Counter(),
            "failed_profiles": Counter(),
            "failed_strategies": Counter(),
            "temporal_patterns": [],
            "correlated_attributes": []
        }
        
        for trace in failed_traces:
            # Count error types
            if trace.error:
                error_type = self._classify_error(trace.error)
                patterns["error_types"][error_type] += 1
            
            # Count failed operations
            patterns["failed_operations"][trace.operation] += 1
            
            # Count failed profiles/strategies
            if trace.profile:
                patterns["failed_profiles"][trace.profile] += 1
            if trace.strategy:
                patterns["failed_strategies"][trace.strategy] += 1
        
        # Analyze temporal patterns
        patterns["temporal_patterns"] = self._analyze_temporal_patterns(
            failed_traces, all_traces
        )
        
        # Find correlated attributes
        patterns["correlated_attributes"] = self._find_correlated_attributes(
            failed_traces, all_traces
        )
        
        return patterns
    
    def _analyze_performance_patterns(
        self,
        slow_traces: List[Any],
        normal_traces: List[Any]
    ) -> Dict[str, Any]:
        """Analyze patterns in performance-degraded traces"""
        patterns = {
            "slow_operations": {},  # Changed from Counter to dict for detailed stats
            "slow_profiles": Counter(),
            "slow_strategies": Counter(),
            "latency_distribution": {},
            "resource_correlation": []
        }
        
        # Collect detailed stats for slow operations
        operation_durations = defaultdict(list)
        for trace in slow_traces:
            operation_durations[trace.operation].append(trace.duration_ms)
            if trace.profile:
                patterns["slow_profiles"][trace.profile] += 1
            if trace.strategy:
                patterns["slow_strategies"][trace.strategy] += 1
        
        # Calculate stats for each slow operation
        for operation, durations in operation_durations.items():
            patterns["slow_operations"][operation] = {
                "count": len(durations),
                "avg_duration": np.mean(durations),
                "min_duration": np.min(durations),
                "max_duration": np.max(durations),
                "durations": durations[:5]  # Sample of actual durations
            }
        
        # Analyze latency distribution
        slow_durations = [t.duration_ms for t in slow_traces]
        normal_durations = [t.duration_ms for t in normal_traces 
                          if t not in slow_traces]
        
        patterns["latency_distribution"] = {
            "slow_mean": np.mean(slow_durations),
            "slow_std": np.std(slow_durations),
            "normal_mean": np.mean(normal_durations) if normal_durations else 0,
            "normal_std": np.std(normal_durations) if normal_durations else 0,
            "slowdown_factor": np.mean(slow_durations) / np.mean(normal_durations) 
                              if normal_durations else 0
        }
        
        return patterns
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type based on error message"""
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        for issue_type, issue_info in self.known_issues.items():
            if re.search(issue_info["pattern"], error_lower):
                return issue_type
        
        # Generic classification
        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower:
            return "memory"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "permission" in error_lower or "forbidden" in error_lower:
            return "permission"
        elif "not found" in error_lower or "404" in error_lower:
            return "not_found"
        else:
            return "unknown"
    
    def _analyze_temporal_patterns(
        self,
        failed_traces: List[Any],
        all_traces: List[Any]
    ) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in failures"""
        patterns = []
        
        # Group by hour
        hourly_failures = defaultdict(int)
        hourly_total = defaultdict(int)
        
        for trace in all_traces:
            hour = trace.timestamp.hour
            hourly_total[hour] += 1
            if trace in failed_traces:
                hourly_failures[hour] += 1
        
        # Find hours with high failure rates
        for hour in hourly_total:
            if hourly_total[hour] > 0:
                failure_rate = hourly_failures[hour] / hourly_total[hour]
                if failure_rate > 0.1:  # More than 10% failure rate
                    patterns.append({
                        "type": "hourly",
                        "hour": hour,
                        "failure_rate": failure_rate,
                        "total_requests": hourly_total[hour],
                        "failed_requests": hourly_failures[hour]
                    })
        
        # Check for failure bursts
        if failed_traces:
            sorted_failures = sorted(failed_traces, key=lambda x: x.timestamp)
            bursts = self._detect_failure_bursts(sorted_failures)
            patterns.extend(bursts)
        
        return patterns
    
    def _detect_failure_bursts(
        self,
        sorted_failures: List[Any],
        burst_window_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Detect bursts of failures within time windows"""
        bursts = []
        
        if len(sorted_failures) < 3:
            return bursts
        
        window = timedelta(minutes=burst_window_minutes)
        i = 0
        
        while i < len(sorted_failures):
            window_start = sorted_failures[i].timestamp
            window_end = window_start + window
            
            # Count failures in window
            window_failures = []
            j = i
            while j < len(sorted_failures) and sorted_failures[j].timestamp <= window_end:
                window_failures.append(sorted_failures[j])
                j += 1
            
            # Check if this is a burst (at least 3 failures in window)
            if len(window_failures) >= 3:
                bursts.append({
                    "type": "burst",
                    "start_time": window_start.isoformat(),
                    "end_time": window_failures[-1].timestamp.isoformat(),
                    "failure_count": len(window_failures),
                    "duration_minutes": (window_failures[-1].timestamp - window_start).total_seconds() / 60,
                    "trace_ids": [t.trace_id for t in window_failures[:5]]  # Sample
                })
                i = j  # Skip past this burst
            else:
                i += 1
        
        return bursts
    
    def _find_correlated_attributes(
        self,
        failed_traces: List[Any],
        all_traces: List[Any]
    ) -> List[FailurePattern]:
        """Find attributes that correlate with failures"""
        correlations = []
        
        # Check profile correlation
        profile_stats = self._calculate_attribute_correlation(
            failed_traces, all_traces, "profile"
        )
        correlations.extend(profile_stats)
        
        # Check strategy correlation
        strategy_stats = self._calculate_attribute_correlation(
            failed_traces, all_traces, "strategy"
        )
        correlations.extend(strategy_stats)
        
        # Check operation correlation
        operation_stats = self._calculate_attribute_correlation(
            failed_traces, all_traces, "operation"
        )
        correlations.extend(operation_stats)
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x.correlation_strength, reverse=True)
        
        return correlations
    
    def _calculate_attribute_correlation(
        self,
        failed_traces: List[Any],
        all_traces: List[Any],
        attribute: str
    ) -> List[FailurePattern]:
        """Calculate correlation between attribute values and failures"""
        patterns = []
        
        # Count occurrences
        failed_values = Counter()
        total_values = Counter()
        
        for trace in all_traces:
            value = getattr(trace, attribute, None)
            if value:
                total_values[value] += 1
                if trace in failed_traces:
                    failed_values[value] += 1
        
        # Calculate failure rates and correlation
        overall_failure_rate = len(failed_traces) / len(all_traces) if all_traces else 0
        
        for value, total_count in total_values.items():
            failed_count = failed_values.get(value, 0)
            failure_rate = failed_count / total_count if total_count > 0 else 0
            
            # Calculate correlation strength (chi-square test)
            if total_count >= 5:  # Minimum sample size
                expected_failures = total_count * overall_failure_rate
                expected_successes = total_count * (1 - overall_failure_rate)
                
                if expected_failures > 0 and expected_successes > 0:
                    chi_square = ((failed_count - expected_failures) ** 2) / expected_failures
                    correlation_strength = min(chi_square / 10, 1.0)  # Normalize
                    
                    # Only include if significant correlation
                    if correlation_strength > 0.3 and failure_rate > overall_failure_rate * 1.5:
                        patterns.append(FailurePattern(
                            pattern_type=attribute,
                            pattern_value=value,
                            failure_rate=failure_rate,
                            occurrence_count=failed_count,
                            confidence=min(correlation_strength * (total_count / 10), 1.0),
                            correlation_strength=correlation_strength
                        ))
        
        return patterns
    
    def _generate_failure_hypotheses(
        self,
        failed_traces: List[Any],
        failure_analysis: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Generate hypotheses for failure root causes"""
        hypotheses = []
        
        # Hypothesis 1: Error type patterns
        if failure_analysis["error_types"]:
            most_common_error = failure_analysis["error_types"].most_common(1)[0]
            error_type, count = most_common_error
            
            if error_type in self.known_issues:
                issue_info = self.known_issues[error_type]
                hypotheses.append(RootCauseHypothesis(
                    hypothesis=f"Primary failure cause: {error_type.replace('_', ' ').title()}",
                    confidence=min(count / len(failed_traces), 0.95),
                    evidence=[
                        f"{count} out of {len(failed_traces)} failures are {error_type}",
                        f"Error pattern matches: {issue_info['pattern']}"
                    ],
                    affected_traces=[t.trace_id for t in failed_traces[:5]],
                    suggested_action=issue_info["suggestion"],
                    category=issue_info["category"]
                ))
        
        # Hypothesis 2: Correlated attributes
        if failure_analysis["correlated_attributes"]:
            for pattern in failure_analysis["correlated_attributes"][:3]:  # Top 3
                hypotheses.append(RootCauseHypothesis(
                    hypothesis=f"{pattern.pattern_type.title()} '{pattern.pattern_value}' has high failure correlation",
                    confidence=pattern.confidence,
                    evidence=[
                        f"Failure rate: {pattern.failure_rate:.1%}",
                        f"Affected traces: {pattern.occurrence_count}",
                        f"Correlation strength: {pattern.correlation_strength:.2f}"
                    ],
                    affected_traces=[],  # Would need to collect these
                    suggested_action=f"Investigate {pattern.pattern_type} '{pattern.pattern_value}' configuration and resources",
                    category="configuration",
                    patterns=[pattern]
                ))
        
        # Hypothesis 3: Temporal patterns
        if failure_analysis["temporal_patterns"]:
            for temporal in failure_analysis["temporal_patterns"]:
                if temporal["type"] == "burst":
                    hypotheses.append(RootCauseHypothesis(
                        hypothesis=f"Failure burst detected at {temporal['start_time']}",
                        confidence=0.8,
                        evidence=[
                            f"{temporal['failure_count']} failures in {temporal['duration_minutes']:.1f} minutes",
                            f"Time range: {temporal['start_time']} to {temporal['end_time']}"
                        ],
                        affected_traces=temporal["trace_ids"],
                        suggested_action="Check for deployment changes or resource issues at this time",
                        category="temporal"
                    ))
        
        return hypotheses
    
    def _generate_performance_hypotheses(
        self,
        slow_traces: List[Any],
        performance_analysis: Dict[str, Any]
    ) -> List[RootCauseHypothesis]:
        """Generate hypotheses for performance degradation"""
        hypotheses = []
        
        # Hypothesis 1: Specific operations are slow
        if performance_analysis["slow_operations"]:
            # Find the operation with the most occurrences
            slowest_op = None
            max_count = 0
            for op_name, op_stats in performance_analysis["slow_operations"].items():
                count = op_stats["count"]
                if count > max_count:
                    max_count = count
                    slowest_op = (op_name, op_stats)
            
            if slowest_op:
                op_name, op_stats = slowest_op
                count = op_stats["count"]
                avg_duration = op_stats["avg_duration"]
                
                hypotheses.append(RootCauseHypothesis(
                    hypothesis=f"Operation '{op_name}' experiencing performance degradation",
                    confidence=min(count / len(slow_traces), 0.9),
                    evidence=[
                        f"{count} slow traces for this operation",
                        f"Average duration: {avg_duration:.1f}ms",
                        f"Duration range: {op_stats['min_duration']:.1f}-{op_stats['max_duration']:.1f}ms",
                        f"Slowdown factor: {performance_analysis['latency_distribution']['slowdown_factor']:.1f}x"
                    ],
                    affected_traces=[t.trace_id for t in slow_traces if t.operation == op_name][:5],
                    suggested_action=f"Optimize '{op_name}' operation or increase resources",
                    category="performance"
                ))
        
        # Hypothesis 2: Profile-specific slowness
        if performance_analysis["slow_profiles"]:
            slowest_profile = performance_analysis["slow_profiles"].most_common(1)[0]
            profile_name, count = slowest_profile
            
            hypotheses.append(RootCauseHypothesis(
                hypothesis=f"Profile '{profile_name}' has performance issues",
                confidence=min(count / len(slow_traces), 0.85),
                evidence=[
                    f"{count} slow traces with this profile",
                    f"Mean latency: {performance_analysis['latency_distribution']['slow_mean']:.0f}ms"
                ],
                affected_traces=[t.trace_id for t in slow_traces if t.profile == profile_name][:5],
                suggested_action=f"Review '{profile_name}' configuration and resource allocation",
                category="configuration"
            ))
        
        return hypotheses
    
    def _perform_statistical_analysis(
        self,
        failed_traces: List[Any],
        successful_traces: List[Any],
        all_traces: List[Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on traces"""
        analysis = {}
        
        if not all_traces:
            return analysis
        
        # Basic statistics
        analysis["failure_rate"] = len(failed_traces) / len(all_traces)
        analysis["success_rate"] = len(successful_traces) / len(all_traces)
        
        # Duration analysis
        if successful_traces:
            success_durations = [t.duration_ms for t in successful_traces]
            analysis["success_duration_stats"] = {
                "mean": np.mean(success_durations),
                "median": np.median(success_durations),
                "std": np.std(success_durations),
                "p95": np.percentile(success_durations, 95)
            }
        
        # Time-based analysis
        analysis["hourly_distribution"] = self._calculate_hourly_distribution(all_traces)
        
        # Profile/strategy comparison
        analysis["profile_comparison"] = self._compare_by_attribute(
            all_traces, failed_traces, "profile"
        )
        analysis["strategy_comparison"] = self._compare_by_attribute(
            all_traces, failed_traces, "strategy"
        )
        
        return analysis
    
    def _calculate_hourly_distribution(self, traces: List[Any]) -> Dict[int, int]:
        """Calculate hourly distribution of traces"""
        hourly_counts = defaultdict(int)
        for trace in traces:
            hourly_counts[trace.timestamp.hour] += 1
        return dict(hourly_counts)
    
    def _compare_by_attribute(
        self,
        all_traces: List[Any],
        failed_traces: List[Any],
        attribute: str
    ) -> Dict[str, Dict[str, Any]]:
        """Compare failure rates by attribute"""
        comparison = {}
        
        # Group traces by attribute
        grouped_all = defaultdict(list)
        grouped_failed = defaultdict(list)
        
        for trace in all_traces:
            value = getattr(trace, attribute, "unknown")
            if value:
                grouped_all[value].append(trace)
                if trace in failed_traces:
                    grouped_failed[value].append(trace)
        
        # Calculate statistics for each group
        for value, traces in grouped_all.items():
            failed = grouped_failed.get(value, [])
            comparison[value] = {
                "total": len(traces),
                "failed": len(failed),
                "failure_rate": len(failed) / len(traces) if traces else 0,
                "mean_duration": np.mean([t.duration_ms for t in traces]),
                "p95_duration": np.percentile([t.duration_ms for t in traces], 95) if traces else 0
            }
        
        return comparison
    
    def _generate_recommendations(
        self,
        root_causes: List[RootCauseHypothesis]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on root causes"""
        recommendations = []
        
        # Group by category
        category_groups = defaultdict(list)
        for hypothesis in root_causes[:5]:  # Top 5 hypotheses
            category_groups[hypothesis.category].append(hypothesis)
        
        # Generate recommendations by category
        for category, hypotheses in category_groups.items():
            if category == "timeout":
                recommendations.append({
                    "priority": "high",
                    "category": category,
                    "recommendation": "Increase timeout settings",
                    "details": [
                        "Review current timeout configurations",
                        "Consider implementing progressive timeouts",
                        "Add retry logic with exponential backoff"
                    ],
                    "affected_components": list(set(h.suggested_action for h in hypotheses))
                })
            
            elif category == "resource":
                recommendations.append({
                    "priority": "high",
                    "category": category,
                    "recommendation": "Scale resources",
                    "details": [
                        "Increase memory allocation",
                        "Review batch sizes",
                        "Consider horizontal scaling"
                    ],
                    "affected_components": list(set(h.suggested_action for h in hypotheses))
                })
            
            elif category == "configuration":
                recommendations.append({
                    "priority": "medium",
                    "category": category,
                    "recommendation": "Review configuration settings",
                    "details": [
                        f"Check {h.hypothesis}" for h in hypotheses
                    ],
                    "affected_components": [h.suggested_action for h in hypotheses]
                })
            
            elif category == "performance":
                recommendations.append({
                    "priority": "medium",
                    "category": category,
                    "recommendation": "Optimize slow operations",
                    "details": [
                        "Profile slow operations",
                        "Add caching where appropriate",
                        "Consider asynchronous processing"
                    ],
                    "affected_components": [h.suggested_action for h in hypotheses]
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
    
    def generate_rca_report(
        self,
        analysis: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """Generate a formatted RCA report"""
        if format == "markdown":
            return self._generate_markdown_report(analysis)
        elif format == "html":
            return self._generate_html_report(analysis)
        else:
            return str(analysis)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown formatted RCA report"""
        report = []
        
        report.append("# Root Cause Analysis Report")
        report.append(f"\n*Generated: {analysis['summary']['analysis_time']}*\n")
        
        # Summary
        report.append("## Summary")
        summary = analysis['summary']
        report.append(f"- Total Traces: {summary['total_traces']}")
        report.append(f"- Failed Traces: {summary['failed_traces']}")
        report.append(f"- Performance Degraded: {summary['performance_degraded']}")
        report.append(f"- Overall Failure Rate: {summary['failure_rate']:.1%}\n")
        
        # Root Causes
        if analysis['root_causes']:
            report.append("## Identified Root Causes\n")
            for i, hypothesis in enumerate(analysis['root_causes'][:5], 1):
                report.append(f"### {i}. {hypothesis.hypothesis}")
                report.append(f"**Confidence:** {hypothesis.confidence:.1%}\n")
                report.append("**Evidence:**")
                for evidence in hypothesis.evidence:
                    report.append(f"- {evidence}")
                report.append(f"\n**Suggested Action:** {hypothesis.suggested_action}\n")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("## Recommendations\n")
            for rec in analysis['recommendations']:
                report.append(f"### {rec['recommendation']} ({rec['priority'].upper()} Priority)")
                report.append("**Actions:**")
                for detail in rec['details']:
                    report.append(f"- {detail}")
                report.append("")
        
        # Statistical Analysis
        if 'statistical_analysis' in analysis:
            stats = analysis['statistical_analysis']
            report.append("## Statistical Analysis\n")
            
            if 'profile_comparison' in stats:
                report.append("### Performance by Profile")
                report.append("| Profile | Total | Failed | Failure Rate | Mean Duration |")
                report.append("|---------|-------|---------|--------------|---------------|")
                for profile, data in stats['profile_comparison'].items():
                    report.append(
                        f"| {profile} | {data['total']} | {data['failed']} | "
                        f"{data['failure_rate']:.1%} | {data['mean_duration']:.0f}ms |"
                    )
                report.append("")
        
        return "\n".join(report)
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML formatted RCA report"""
        html = f"""
        <html>
        <head>
            <title>Root Cause Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #d32f2f; }}
                h2 {{ color: #f57c00; }}
                h3 {{ color: #388e3c; }}
                .confidence {{ 
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 4px;
                    background: #e3f2fd;
                    color: #1976d2;
                    font-weight: bold;
                }}
                .priority-high {{ color: #d32f2f; font-weight: bold; }}
                .priority-medium {{ color: #f57c00; font-weight: bold; }}
                .priority-low {{ color: #388e3c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .evidence {{ background: #fff3e0; padding: 10px; border-left: 3px solid #ff9800; }}
                .recommendation {{ background: #e8f5e9; padding: 10px; border-left: 3px solid #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Root Cause Analysis Report</h1>
            <p><em>Generated: {analysis['summary']['analysis_time']}</em></p>
            
            <h2>Summary</h2>
            <ul>
                <li>Total Traces: {analysis['summary']['total_traces']}</li>
                <li>Failed Traces: {analysis['summary']['failed_traces']}</li>
                <li>Performance Degraded: {analysis['summary']['performance_degraded']}</li>
                <li>Overall Failure Rate: {analysis['summary']['failure_rate']:.1%}</li>
            </ul>
        """
        
        # Add root causes
        if analysis['root_causes']:
            html += "<h2>Identified Root Causes</h2>"
            for i, hypothesis in enumerate(analysis['root_causes'][:5], 1):
                confidence_pct = hypothesis.confidence * 100
                html += f"""
                <h3>{i}. {hypothesis.hypothesis}</h3>
                <p><span class="confidence">Confidence: {confidence_pct:.0f}%</span></p>
                <div class="evidence">
                    <strong>Evidence:</strong>
                    <ul>
                """
                for evidence in hypothesis.evidence:
                    html += f"<li>{evidence}</li>"
                html += f"""
                    </ul>
                    <strong>Suggested Action:</strong> {hypothesis.suggested_action}
                </div>
                """
        
        # Add recommendations
        if analysis['recommendations']:
            html += "<h2>Recommendations</h2>"
            for rec in analysis['recommendations']:
                priority_class = f"priority-{rec['priority']}"
                html += f"""
                <div class="recommendation">
                    <h3 class="{priority_class}">{rec['recommendation']}</h3>
                    <ul>
                """
                for detail in rec['details']:
                    html += f"<li>{detail}</li>"
                html += "</ul></div>"
        
        html += "</body></html>"
        return html