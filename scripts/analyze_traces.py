#!/usr/bin/env python3
"""
Analyze Phoenix traces and generate comprehensive visualizations

This script provides analytics and visualization capabilities for traces collected
during Cogniverse evaluations, including request statistics, response time analysis,
and outlier detection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.phoenix.analytics import PhoenixAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TraceAnalyzer:
    """Interactive trace analyzer with various visualization options"""
    
    def __init__(self, phoenix_url: str = "http://localhost:6006"):
        self.analytics = PhoenixAnalytics(phoenix_url)
        self.output_dir = Path("outputs/analytics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_time_period(
        self,
        hours: float = 24,
        operation_filter: Optional[str] = None,
        save_plots: bool = True
    ):
        """
        Analyze traces from the last N hours
        
        Args:
            hours: Number of hours to look back
            operation_filter: Filter by operation name
            save_plots: Whether to save plots to files
        """
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        logger.info(f"Analyzing traces from {start_time} to {end_time}")
        
        # Fetch traces
        traces = self.analytics.get_traces(
            start_time=start_time,
            end_time=end_time,
            operation_filter=operation_filter
        )
        
        if not traces:
            logger.warning("No traces found in the specified time period")
            return
        
        logger.info(f"Found {len(traces)} traces")
        
        # Calculate statistics
        stats = self.analytics.calculate_statistics(traces)
        stats_by_profile = self.analytics.calculate_statistics(traces, group_by="profile")
        stats_by_operation = self.analytics.calculate_statistics(traces, group_by="operation")
        
        # Print summary
        self._print_summary(stats)
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Time series plot
        time_series_fig = self.analytics.create_time_series_plot(
            traces, metric="duration_ms", aggregation="mean", time_window="5min"
        )
        if save_plots:
            plot_file = self.output_dir / f"time_series_{timestamp}.html"
            time_series_fig.write_html(str(plot_file))
            logger.info(f"Time series plot saved to: {plot_file}")
        else:
            time_series_fig.show()
        
        # Distribution plot
        distribution_fig = self.analytics.create_distribution_plot(
            traces, metric="duration_ms", group_by="profile"
        )
        if save_plots:
            plot_file = self.output_dir / f"distribution_{timestamp}.html"
            distribution_fig.write_html(str(plot_file))
            logger.info(f"Distribution plot saved to: {plot_file}")
        else:
            distribution_fig.show()
        
        # Outlier plot
        outlier_fig = self.analytics.create_outlier_plot(traces)
        if save_plots:
            plot_file = self.output_dir / f"outliers_{timestamp}.html"
            outlier_fig.write_html(str(plot_file))
            logger.info(f"Outlier plot saved to: {plot_file}")
        else:
            outlier_fig.show()
        
        # Heatmap
        heatmap_fig = self.analytics.create_heatmap(
            traces, x_field="hour", y_field="day", 
            metric="duration_ms", aggregation="mean"
        )
        if save_plots:
            plot_file = self.output_dir / f"heatmap_{timestamp}.html"
            heatmap_fig.write_html(str(plot_file))
            logger.info(f"Heatmap saved to: {plot_file}")
        else:
            heatmap_fig.show()
        
        # Comparison plot
        comparison_fig = self.analytics.create_comparison_plot(
            traces, compare_field="profile", metric="duration_ms"
        )
        if save_plots:
            plot_file = self.output_dir / f"comparison_{timestamp}.html"
            comparison_fig.write_html(str(plot_file))
            logger.info(f"Comparison plot saved to: {plot_file}")
        else:
            comparison_fig.show()
        
        # Save full report
        report_file = self.output_dir / f"report_{timestamp}.json"
        report = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "statistics": stats,
            "statistics_by_profile": stats_by_profile,
            "statistics_by_operation": stats_by_operation
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Full report saved to: {report_file}")
    
    def _print_summary(self, stats: dict):
        """Print formatted statistics summary"""
        print("\n" + "="*60)
        print("TRACE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal Requests: {stats.get('total_requests', 0)}")
        
        if "time_range" in stats:
            time_range = stats["time_range"]
            print(f"Time Range: {time_range.get('start', 'N/A')} to {time_range.get('end', 'N/A')}")
        
        if "response_time" in stats:
            rt = stats["response_time"]
            print(f"\nResponse Time Statistics (ms):")
            print(f"  Mean: {rt.get('mean', 0):.2f}")
            print(f"  Median: {rt.get('median', 0):.2f}")
            print(f"  Min: {rt.get('min', 0):.2f}")
            print(f"  Max: {rt.get('max', 0):.2f}")
            print(f"  Std Dev: {rt.get('std', 0):.2f}")
            print(f"  P50: {rt.get('p50', 0):.2f}")
            print(f"  P75: {rt.get('p75', 0):.2f}")
            print(f"  P90: {rt.get('p90', 0):.2f}")
            print(f"  P95: {rt.get('p95', 0):.2f}")
            print(f"  P99: {rt.get('p99', 0):.2f}")
        
        if "status" in stats:
            status = stats["status"]
            print(f"\nStatus Distribution:")
            for status_type, count in status.get("counts", {}).items():
                print(f"  {status_type}: {count}")
            print(f"  Success Rate: {status.get('success_rate', 0):.2%}")
            print(f"  Error Rate: {status.get('error_rate', 0):.2%}")
        
        if "outliers" in stats:
            outliers = stats["outliers"]
            print(f"\nOutliers:")
            print(f"  Count: {outliers.get('count', 0)}")
            print(f"  Percentage: {outliers.get('percentage', 0):.2f}%")
            if outliers.get('values'):
                outlier_values = outliers['values'][:5]  # Show first 5
                print(f"  Sample Values (ms): {outlier_values}")
        
        if "temporal" in stats:
            temporal = stats["temporal"]
            if "requests_by_hour" in temporal:
                print(f"\nRequests by Hour:")
                for hour, count in sorted(temporal["requests_by_hour"].items()):
                    print(f"  Hour {hour}: {count} requests")
    
    def monitor_realtime(
        self,
        refresh_interval: int = 30,
        window_minutes: int = 60
    ):
        """
        Monitor traces in real-time with periodic updates
        
        Args:
            refresh_interval: Seconds between updates
            window_minutes: Time window to analyze in minutes
        """
        logger.info(f"Starting real-time monitoring (refresh every {refresh_interval}s)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                # Clear screen (works on Unix-like systems)
                print("\033[2J\033[H")
                
                # Calculate time window
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=window_minutes)
                
                # Fetch recent traces
                traces = self.analytics.get_traces(
                    start_time=start_time,
                    end_time=end_time
                )
                
                if traces:
                    # Calculate statistics
                    stats = self.analytics.calculate_statistics(traces)
                    
                    # Display dashboard
                    print("="*60)
                    print(f"REAL-TIME MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Window: Last {window_minutes} minutes")
                    print("="*60)
                    
                    self._print_summary(stats)
                    
                    # Show recent traces
                    print(f"\nRecent Traces (last 5):")
                    for trace in traces[-5:]:
                        print(f"  {trace.timestamp.strftime('%H:%M:%S')} - "
                              f"{trace.operation} - "
                              f"{trace.duration_ms:.2f}ms - "
                              f"{trace.status}")
                else:
                    print("No traces found in the time window")
                
                print(f"\nNext refresh in {refresh_interval} seconds...")
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
    
    def generate_comprehensive_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_format: str = "all"
    ):
        """
        Generate comprehensive analytics report
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            output_format: Output format (json, html, all)
        """
        # Default to last 24 hours if not specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=24)
        
        logger.info(f"Generating comprehensive report from {start_time} to {end_time}")
        
        # Generate report using analytics module
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format in ["json", "all"]:
            json_file = self.output_dir / f"comprehensive_report_{timestamp}.json"
            report = self.analytics.generate_report(
                start_time=start_time,
                end_time=end_time,
                output_file=str(json_file)
            )
            logger.info(f"JSON report saved to: {json_file}")
        
        if output_format in ["html", "all"]:
            # Generate HTML report with embedded visualizations
            html_file = self.output_dir / f"comprehensive_report_{timestamp}.html"
            self._generate_html_report(start_time, end_time, html_file)
            logger.info(f"HTML report saved to: {html_file}")
        
        print(f"\nReports generated successfully in: {self.output_dir}")
    
    def _generate_html_report(
        self,
        start_time: datetime,
        end_time: datetime,
        output_file: Path
    ):
        """Generate HTML report with embedded visualizations"""
        # Fetch traces
        traces = self.analytics.get_traces(start_time=start_time, end_time=end_time)
        
        if not traces:
            logger.warning("No traces found for HTML report")
            return
        
        # Generate all visualizations
        time_series_fig = self.analytics.create_time_series_plot(traces)
        distribution_fig = self.analytics.create_distribution_plot(traces)
        outlier_fig = self.analytics.create_outlier_plot(traces)
        heatmap_fig = self.analytics.create_heatmap(traces)
        comparison_fig = self.analytics.create_comparison_plot(traces)
        
        # Calculate statistics
        stats = self.analytics.calculate_statistics(traces)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phoenix Trace Analytics Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #333;
        }}
        .plot-container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>Phoenix Trace Analytics Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    <p>Time Range: {start_time.isoformat()} to {end_time.isoformat()}</p>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <div class="metric">
            <div class="metric-label">Total Requests</div>
            <div class="metric-value">{stats.get('total_requests', 0):,}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean Response Time</div>
            <div class="metric-value">{stats.get('response_time', {}).get('mean', 0):.2f} ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">P95 Response Time</div>
            <div class="metric-value">{stats.get('response_time', {}).get('p95', 0):.2f} ms</div>
        </div>
        <div class="metric">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">{stats.get('status', {}).get('success_rate', 0):.1%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Outliers</div>
            <div class="metric-value">{stats.get('outliers', {}).get('percentage', 0):.1f}%</div>
        </div>
    </div>
    
    <div class="plot-container">
        <h2>Response Time Over Time</h2>
        {time_series_fig.to_html(include_plotlyjs='cdn', div_id="time-series")}
    </div>
    
    <div class="plot-container">
        <h2>Response Time Distribution</h2>
        {distribution_fig.to_html(include_plotlyjs=False, div_id="distribution")}
    </div>
    
    <div class="plot-container">
        <h2>Outlier Detection</h2>
        {outlier_fig.to_html(include_plotlyjs=False, div_id="outliers")}
    </div>
    
    <div class="plot-container">
        <h2>Request Heatmap</h2>
        {heatmap_fig.to_html(include_plotlyjs=False, div_id="heatmap")}
    </div>
    
    <div class="plot-container">
        <h2>Profile Comparison</h2>
        {comparison_fig.to_html(include_plotlyjs=False, div_id="comparison")}
    </div>
    
    <div class="footer">
        <p>Cogniverse Evaluation Framework - Phoenix Analytics</p>
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Phoenix traces and generate visualizations"
    )
    
    parser.add_argument(
        "--phoenix-url",
        default="http://localhost:6006",
        help="Phoenix server URL (default: http://localhost:6006)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Analysis commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze traces from time period")
    analyze_parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Number of hours to look back (default: 24)"
    )
    analyze_parser.add_argument(
        "--operation",
        help="Filter by operation name"
    )
    analyze_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show plots instead of saving to files"
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Real-time monitoring")
    monitor_parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)"
    )
    monitor_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Time window in minutes (default: 60)"
    )
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument(
        "--start",
        help="Start time (ISO format)"
    )
    report_parser.add_argument(
        "--end",
        help="End time (ISO format)"
    )
    report_parser.add_argument(
        "--format",
        choices=["json", "html", "all"],
        default="all",
        help="Output format (default: all)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create analyzer
    analyzer = TraceAnalyzer(phoenix_url=args.phoenix_url)
    
    # Execute command
    try:
        if args.command == "analyze":
            analyzer.analyze_time_period(
                hours=args.hours,
                operation_filter=args.operation,
                save_plots=not args.no_save
            )
        
        elif args.command == "monitor":
            analyzer.monitor_realtime(
                refresh_interval=args.refresh,
                window_minutes=args.window
            )
        
        elif args.command == "report":
            start_time = datetime.fromisoformat(args.start) if args.start else None
            end_time = datetime.fromisoformat(args.end) if args.end else None
            analyzer.generate_comprehensive_report(
                start_time=start_time,
                end_time=end_time,
                output_format=args.format
            )
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()