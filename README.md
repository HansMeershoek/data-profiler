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
- 🎯 **Target Analysis**: Special insights for classification/regression tasks
- 🔍 **Comprehensive Profiling**: Detailed statistics and distributions
- ⚡ **Performance Optimized**: Efficient handling of large datasets
- 🛠️ **Customizable**: Configure sections and visualization options

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
from pytics import profile

# Load your dataset
df = pd.read_csv('your_data.csv')

# Generate an HTML report
profile(df, output_file='report.html')

# Generate a PDF report
profile(df, output_format='pdf', output_file='report.pdf')

# Profile with a target variable
profile(df, target='target_column', output_file='report.html')

# Select specific sections
profile(
    df,
    include_sections=['overview', 'correlations'],
    output_file='report.html'
)
```

## Report Sections

1. **Overview**
   - Dataset summary
   - Memory usage
   - Data types distribution
   - Missing values summary

2. **Variable Analysis**
   - Detailed statistics
   - Distribution plots
   - Missing value patterns
   - Unique values analysis

3. **Correlations**
   - Correlation matrix
   - Feature relationships
   - Interactive heatmaps

4. **Target Analysis** (when target specified)
   - Target distribution
   - Feature importance
   - Target correlations

## Configuration Options

```python
profile(
    df,
    target='target_column',           # Target variable for supervised learning
    include_sections=['overview'],    # Sections to include
    exclude_sections=['correlations'],# Sections to exclude
    output_format='pdf',             # 'html' or 'pdf'
    output_file='report.html',       # Output file path
    theme='light',                   # Report theme
    title='Custom Report Title'      # Report title
)
```

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
