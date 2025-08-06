# Phoenix Analytics Dashboard

A comprehensive, real-time visualization dashboard for analyzing Phoenix traces collected during Cogniverse evaluations.

## Features

### 🎯 Real-Time Analytics
- **Auto-refresh**: Configurable refresh intervals (5-300 seconds)
- **Live metrics**: Continuously updated statistics and visualizations
- **Time range selection**: Quick presets or custom date/time ranges

### 📊 Comprehensive Visualizations

#### Overview Tab
- Key performance metrics (total requests, response times, success rates)
- Response time statistics table
- Status distribution pie chart
- Performance comparison by profile

#### Time Series Tab
- Response time trends with configurable aggregation windows
- Percentile bands (P50, P95)
- Request rate over time
- Multiple aggregation methods (mean, median, max, min, count)

#### Distribution Tab
- 4-panel distribution analysis (histogram, box plot, violin plot, ECDF)
- Group-by capabilities (profile, strategy, operation)
- Comparative analysis charts

#### Heatmap Tab
- Customizable axes (hour, day, profile, strategy)
- Multiple metrics (response time, request count)
- Various aggregation methods

#### Outlier Tab
- Visual outlier detection with threshold lines
- Outlier statistics and percentages
- Detailed outlier trace table
- Interactive scatter plot with hover details

#### Trace Explorer Tab
- Search and filter capabilities
- Sortable columns
- Pagination for large datasets
- Status-based coloring

### 🔍 Advanced Filtering
- Operation name filtering
- Profile multi-select
- Strategy multi-select
- Time range selection
- Status filtering

### 📥 Export Capabilities
- JSON reports with full statistics
- CSV exports of trace data
- HTML reports for sharing

## Installation

```bash
# Install Streamlit and dependencies
uv pip install streamlit

# Or sync from pyproject.toml
uv sync
```

## Usage

### Starting the Dashboard

```bash
# Basic usage
uv run streamlit run scripts/phoenix_dashboard_standalone.py

# With custom port
uv run streamlit run scripts/phoenix_dashboard_standalone.py --server.port 8501

# Note: Phoenix URL is hardcoded to http://localhost:6006
```

### Access the Dashboard

Once started, the dashboard will be available at:
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

## Dashboard Interface

### Sidebar Configuration

1. **Phoenix Connection**
   - Configure Phoenix server URL
   - Test connection button

2. **Time Range**
   - Quick presets: Last Hour, 6 Hours, 24 Hours, 7 Days
   - Custom date/time selection

3. **Filters**
   - Operation name filter
   - Profile multi-select
   - Strategy multi-select

4. **Auto Refresh**
   - Enable/disable toggle
   - Refresh interval slider
   - Progress indicator

5. **Export**
   - JSON, CSV, or HTML format
   - Download buttons

### Main Content Area

The dashboard has two main tabs:

#### 📊 Analytics Tab
Contains 6 sub-tabs (7 with RCA enabled):
1. **Overview**: High-level metrics and summary statistics
2. **Time Series**: Temporal analysis and trends
3. **Distributions**: Statistical distributions and comparisons
4. **Heatmaps**: 2D visualization of metrics
5. **Outliers**: Anomaly detection and analysis
6. **Trace Explorer**: Detailed trace inspection
7. **Root Cause Analysis** (optional): Automated failure diagnosis

#### 🧪 Evaluation Tab
Displays Phoenix experiment results:
- **Dataset Selection**: Choose from all available Phoenix datasets
- **Automatic Experiment Loading**: Uses `/v1/datasets/{id}/experiments` API
- **Profile/Strategy Tabs**: Dynamic nested tabs based on actual experiments run
- **Query Results Table**: Horizontal layout with:
  - Query text
  - Expected videos
  - Retrieved results with ✅/❌ evaluation
  - Colored metric badges (MRR, R@1, R@5)
- **Success Matrix Heatmap**: Visual overview of experiment performance

## Key Features

### Auto-Refresh
The dashboard supports automatic refresh with visual countdown:
```python
# Enabled in sidebar
✓ Enable Auto Refresh
Refresh Interval: 30 seconds
[Progress bar showing time until next refresh]
```

### Interactive Visualizations
All plots are interactive with:
- Zoom and pan
- Hover tooltips
- Legend toggling
- Export to PNG

### Performance Optimization
- Efficient trace fetching with limits
- Client-side filtering for responsiveness
- Pagination for large datasets
- Caching of analytics results

## Use Cases

### 1. Real-Time Monitoring
Monitor ongoing evaluations with auto-refresh:
- Track request rates
- Watch for performance degradation
- Identify errors as they occur

### 2. Post-Evaluation Analysis
Analyze completed evaluations:
- Compare profile performance
- Identify outliers and anomalies
- Generate reports for stakeholders

### 3. Troubleshooting
Debug issues with detailed trace inspection:
- Filter by error status
- Search for specific operations
- Examine outlier traces

### 4. Performance Optimization
Identify optimization opportunities:
- Find slow operations
- Compare strategy effectiveness
- Analyze temporal patterns

## Tips and Best Practices

1. **Start with Overview**: Get high-level insights before diving into details
2. **Use Filters**: Narrow down to specific profiles/strategies for focused analysis
3. **Check Outliers**: Always investigate outliers for potential issues
4. **Export Reports**: Save snapshots for documentation and comparison
5. **Monitor Trends**: Use time series to spot degradation over time

## Troubleshooting

### Dashboard won't start
```bash
# Check if Streamlit is installed
uv pip list | grep streamlit

# Install if missing
uv pip install streamlit
```

### Can't connect to Phoenix
1. Verify Phoenix is running: `curl http://localhost:6006/health`
2. Check the Phoenix URL in sidebar configuration
3. Use "Test Connection" button to verify

### Slow performance
1. Reduce time range to fetch fewer traces
2. Increase refresh interval for auto-refresh
3. Use filters to reduce data volume

### Missing data
1. Verify traces exist in Phoenix
2. Check time range covers evaluation period
3. Remove filters that might be too restrictive

## Keyboard Shortcuts

- `R`: Refresh now
- `F`: Toggle fullscreen
- `Esc`: Exit fullscreen
- `?`: Show help (when implemented)

## Configuration

The dashboard can be configured via Streamlit config:

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

## Root Cause Analysis (RCA)

The dashboard includes a powerful Root Cause Analysis feature that automatically diagnoses failures and performance issues.

### What is Root Cause Analysis?

RCA automatically identifies:
- **Why** certain requests failed or were slow
- **What patterns** lead to errors
- **Which components** are problematic
- **Correlations** between failures and specific conditions

### Enabling RCA

1. In the sidebar, find "🔍 Root Cause Analysis"
2. Check "Enable RCA"
3. Configure options:
   - **Include Performance Analysis**: Analyze slow traces too
   - **Performance Threshold Percentile**: Define what's considered "slow" (default: 95th percentile)
4. A new "🎯 Root Cause Analysis" tab appears

### RCA Features

#### 1. **Automated Failure Analysis**
- **Error Classification**: Automatically categorizes errors (timeout, memory, network, rate limit, etc.)
- **Pattern Detection**: Finds correlations between failures and attributes (profile, strategy, operation)
- **Temporal Analysis**: Detects failure bursts and time-based patterns
- **Statistical Correlation**: Uses chi-square tests to find significant patterns

#### 2. **Performance Degradation Analysis**
- Identifies traces slower than the configured percentile
- Calculates slowdown factors (e.g., "3.2x slower than normal")
- Compares slow vs normal performance distributions
- Pinpoints operations/profiles with performance issues

#### 3. **Root Cause Hypotheses**
Each hypothesis includes:
- **Confidence Score** (0-100%): Statistical confidence in the hypothesis
- **Evidence**: Specific metrics and patterns supporting the hypothesis
- **Affected Traces**: Sample trace IDs for investigation
- **Suggested Actions**: Specific steps to remediate the issue
- **Category**: Type of issue (timeout, resource, configuration, etc.)

#### 4. **Prioritized Recommendations**
- 🔴 **High Priority**: Critical issues requiring immediate attention
- 🟡 **Medium Priority**: Important but not critical issues
- 🟢 **Low Priority**: Optimizations and improvements
- Each recommendation includes specific action items

### Using RCA

1. **Run Analysis**:
   ```
   Click "🔍 Run Analysis" button
   ```

2. **Review Results**:
   - **Summary Metrics**: Failed traces, failure rate, performance degraded count
   - **Root Causes**: Ranked by confidence with expandable details
   - **Failure Patterns**: Visual analysis of error types and distributions
   - **Temporal Patterns**: Burst detection and hourly patterns
   - **Recommendations**: Actionable steps to fix issues

3. **Export Report**:
   ```
   Click "📥 Export RCA Report" to download markdown report
   ```

### Example RCA Output

```
🎯 Identified Root Causes:

1. Primary failure cause: Timeout (95% confidence)
   Evidence:
   - 45 out of 50 failures are timeout
   - Error pattern matches: (timeout|timed out)
   Suggested Action: Increase timeout limits or optimize query processing

2. Profile 'direct_video_global' has high failure correlation (82% confidence)
   Evidence:
   - Failure rate: 35.2%
   - Affected traces: 28
   - Correlation strength: 0.75
   Suggested Action: Investigate profile configuration and resources

⚠️ Failure Burst Detected:
   12 failures in 2.3 minutes starting at 2024-01-15 14:30:00
   
💡 High Priority Recommendation:
   Increase timeout settings
   - Review current timeout configurations
   - Consider implementing progressive timeouts
   - Add retry logic with exponential backoff
```

### RCA for Different Scenarios

#### Timeout Issues
- Detects patterns like "timeout", "deadline exceeded"
- Recommends: Timeout increases, query optimization, batch size reduction

#### Memory Issues
- Detects "out of memory", "OOM", "allocation failed"
- Recommends: Memory increase, batch size reduction, memory profiling

#### Network Issues
- Detects "connection refused", "network error"
- Recommends: Service health checks, retry logic, circuit breakers

#### Performance Degradation
- Identifies operations running slower than baseline
- Recommends: Caching, async processing, resource scaling

### Customizing RCA

The RCA system is extensible. You can add custom patterns:

```python
# In root_cause_analysis.py
self.known_issues = {
    "custom_error": {
        "pattern": r"(your custom pattern)",
        "category": "custom",
        "suggestion": "Your custom suggestion"
    }
}
```

### RCA Best Practices

1. **Regular Analysis**: Run RCA after deployments or when issues arise
2. **Track Trends**: Compare RCA results over time to see improvements
3. **Act on High Priority**: Address high-priority recommendations first
4. **Export Reports**: Save RCA reports for documentation and tracking
5. **Combine with Monitoring**: Use alongside real-time monitoring for best results

## Future Enhancements

Planned features for future versions:
- [ ] Alert notifications for threshold breaches
- [ ] Saved dashboard configurations
- [ ] Comparative analysis between evaluations
- [ ] Custom metric definitions
- [ ] Embedded in main application
- [ ] Dark mode support
- [ ] Mobile-responsive design
- [ ] Real-time collaborative features
- [ ] ML-based root cause prediction
- [ ] Integration with incident management systems

## Integration with CI/CD

The dashboard can be deployed as part of CI/CD:

```yaml
# .github/workflows/dashboard.yml
name: Deploy Dashboard

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to Streamlit Cloud
        run: |
          # Deploy to Streamlit Cloud
          streamlit deploy scripts/phoenix_dashboard.py
```

## API Reference

The dashboard uses the `PhoenixAnalytics` class internally:

```python
from src.evaluation.phoenix.analytics import PhoenixAnalytics

# Used internally by dashboard
analytics = PhoenixAnalytics(phoenix_url)
traces = analytics.get_traces(start_time, end_time)
stats = analytics.calculate_statistics(traces)
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Phoenix logs: `tail -f data/phoenix/phoenix.log`
3. Check Streamlit logs in terminal
4. File an issue with error details and screenshots