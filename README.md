# pytics

[![PyPI version](https://img.shields.io/pypi/v/pytics)](https://pypi.org/project/pytics/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pytics)](https://pypi.org/project/pytics/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/HansMeershoek/pytics/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/HansMeershoek/pytics/actions/workflows/python-test.yml)

An interactive data profiling library for Python that generates comprehensive HTML reports with rich visualizations and PDF export capabilities.

## Features

- 📊 **Interactive Visualizations**: Built with Plotly for dynamic, interactive charts
- 📱 **Responsive Design**: Reports adapt to different screen sizes
- 📄 **PDF Export**: Generate publication-ready PDF reports
- 📋 **JSON Export**: Export profiling data in structured JSON format
- 🎯 **Target Analysis**: Special insights for classification/regression tasks
- 🔍 **Comprehensive Profiling**: Detailed statistics and distributions
- ⚡ **Performance Optimized**: Efficient handling of large datasets
- 🛠️ **Customizable**: Configure sections and visualization options
- ↔️ **DataFrame Comparison**: Compare two datasets for differences in schema, stats, and distributions

## Example Reports

### Full Profile Report
![Full Profile Report](examples/full_report.png)

### Targeted Analysis Report
![Targeted Analysis Report](examples/targeted_report.png)

## Installation

```bash
pip install pytics
```

## Quick Start

```python
import pandas as pd
from pytics import profile, compare

# --- Basic Profiling ---
# Method 1: Profile a DataFrame object
df = pd.read_csv('your_data.csv')
profile(df, output_file='report.html')

# Method 2: Profile directly from a file path
# Supports CSV and Parquet files
profile('path/to/your_data.csv', output_file='report.html')
profile('path/to/your_data.parquet', output_file='report.html')

# --- Advanced Profiling ---
# Generate a PDF report
profile(df, output_format='pdf', output_file='report.pdf')

# Generate a JSON report
profile(df, output_format='json', output_file='report.json')

# Profile with a target variable for enhanced analysis
profile(
    df,
    target='target_column',  # Enables target-specific analysis
    output_file='targeted_report.html'
)

# Select specific sections to include/exclude
profile(
    df,
    include_sections=['overview', 'correlations'],
    exclude_sections=['target_analysis'],
    output_file='custom_report.html'
)

# --- DataFrame Comparison ---
# Method 1: Compare two DataFrame objects
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

compare(
    df_train, 
    df_test,
    name1='Train Set',    # Optional: Custom names for the datasets
    name2='Test Set',
    output_file='comparison.html'
)

# Method 2: Compare directly from file paths
compare(
    'path/to/train_data.csv',
    'path/to/test_data.csv',
    name1='Train Set',
    name2='Test Set',
    output_file='comparison.html'
)
```

## Target Variable Analysis

When you specify a target variable using the `target` parameter, pytics enhances the analysis with:

- Target distribution visualization
- Feature importance analysis
- Target-specific correlations
- Conditional distributions of features
- Statistical tests for feature-target relationships

Example:
```python
# Profile with target variable analysis
profile(
    df,
    target='target_column',
    output_file='targeted_report.html'
)
```

## Configuration Options

### Profile Configuration
```python
profile(
    df,
    target='target_column',           # Target variable for supervised learning
    include_sections=['overview'],    # Sections to include
    exclude_sections=['correlations'],# Sections to exclude
    output_format='pdf',             # 'html', 'pdf', or 'json'
    output_file='report.html',       # Output file path
    theme='light',                   # Report theme ('light' or 'dark')
    title='Custom Report Title'      # Report title
)
```

### Compare Configuration
```python
compare(
    df1,
    df2,
    name1='First Dataset',           # Custom name for first dataset
    name2='Second Dataset',          # Custom name for second dataset
    output_file='comparison.html',   # Output file path
    theme='light',                   # Report theme ('light' or 'dark')
    title='Dataset Comparison'       # Report title
)
```

### JSON Export
The JSON export format provides a structured representation of the profiling data, suitable for programmatic analysis or integration with other tools. The JSON output includes:

```python
{
    "metadata": {
        "title": "Report Title",
        "generated_at": "2024-01-01T00:00:00",
        "pytics_version": "1.1.4",
        "schema_version": "1.0"
    },
    "overview": {
        "shape": {"rows": 1000, "columns": 10},
        "n_vars": 10,
        "n_obs": 1000,
        "memory_usage": "1.2 MB",
        ...
    },
    "variables": {
        "column_name": {
            "type": "float64",
            "missing_count": 0,
            "statistics": {
                "mean": 0.5,
                "std": 0.2,
                ...
            },
            "distribution": {
                "type": "histogram",
                "counts": [...],
                "bin_edges": [...]
            }
        },
        ...
    },
    "correlations": {...},
    "missing_values": {...},
    "duplicates": {...},
    "target_analysis": {...}  # When target is specified
}
```

Example usage:
```python
# Generate JSON report
profile(df, output_format='json', output_file='report.json')

# Load and use the JSON data
import json
with open('report.json', 'r') as f:
    profile_data = json.load(f)

# Access specific metrics
n_missing = profile_data['overview']['n_missing']
column_stats = profile_data['variables']['column_name']['statistics']
```

### Available Sections
- `overview`: Dataset summary and memory usage
- `variables`: Detailed variable analysis
- `correlations`: Correlation analysis
- `target_analysis`: Target-specific insights (requires target parameter)
- `interactions`: Feature interaction analysis
- `missing_values`: Missing value patterns
- `duplicates`: Duplicate record analysis

## Report Sections

1. **Overview**
   - Dataset summary
   - Memory usage
   - Data types distribution
   - Missing values summary

2. **DataFrame Summary**
   - Complete DataFrame info output
   - Numerical and categorical statistics
   - Data preview (head/tail)
   - Memory usage details

3. **Variable Analysis**
   - Detailed statistics
   - Distribution plots
   - Missing value patterns
   - Unique values analysis

4. **Correlations**
   - Correlation matrix
   - Feature relationships
   - Interactive heatmaps

5. **Target Analysis** (when target specified)
   - Target distribution
   - Feature importance
   - Target correlations

6. **Missing Values**
   - Missing value patterns
   - Distribution analysis
   - Correlation with other features

7. **Duplicates**
   - Duplicate record analysis
   - Pattern identification
   - Impact assessment

8. **About**
   - Project information
   - Feature overview
   - GitHub repository links

## Edge Cases and Limitations

### Data Size Limits
- Recommended maximum rows: 1 million
- Recommended maximum columns: 1000
- Large datasets may require increased memory allocation

### Special Cases
- Missing Values: Automatically handled and reported
- Categorical Variables: Limited to 1000 unique values by default
- Date/Time: Automatically detected and analyzed
- Mixed Data Types: Handled with appropriate warnings

### Error Handling
- Custom exceptions for clear error reporting
- Warning system for non-critical issues
- Graceful degradation for memory constraints

## Best Practices

1. **Memory Management**
   - Sample large datasets if needed
   - Use section selection for focused analysis
   - Monitor memory usage for big datasets

2. **Performance Optimization**
   - Limit categorical variables when possible
   - Use targeted section selection
   - Consider data sampling for initial exploration

3. **Report Generation**
   - Choose appropriate output format
   - Use meaningful report titles
   - Save reports with descriptive filenames

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
