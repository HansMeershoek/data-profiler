"""
Tests for the data profiler functionality
"""
import pytest
import pandas as pd
import numpy as np
from pytics import profile
from pytics.profiler import DataSizeError, ProfilerError, compare
from pathlib import Path
from jinja2 import Environment, PackageLoader
import builtins
import json
from datetime import datetime, timedelta
from jsonschema import validate, ValidationError
from .json_schema import PYTICS_JSON_SCHEMA

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'numeric': np.random.normal(0, 1, n_samples),
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples),
        'missing': np.where(np.random.random(n_samples) > 0.7, np.nan, np.random.random(n_samples))
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_df2():
    """Create a second sample DataFrame for comparison testing"""
    np.random.seed(43)  # Different seed from sample_df
    n_samples = 100
    
    data = {
        'numeric': np.random.normal(0, 1, n_samples),  # Same column name, same type
        'categorical': np.random.choice(['X', 'Y', 'Z'], n_samples),  # Same column name, same type
        'new_column': np.random.random(n_samples),  # New column not in df1
        'different_type': pd.Series(np.random.choice(['A', 'B'], n_samples)).astype('category')  # Will be compared with 'target' which is int
    }
    return pd.DataFrame(data)

@pytest.fixture
def complex_df():
    """Create a DataFrame with various data types for testing JSON export"""
    np.random.seed(42)
    n_samples = 100
    
    # Create datetime range
    date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    data = {
        'int_col': np.random.randint(-100, 100, n_samples),
        'float_col': np.random.normal(0, 1, n_samples),
        'bool_col': np.random.choice([True, False], n_samples),
        'cat_col': pd.Categorical(np.random.choice(['A', 'B', 'C'], n_samples)),
        'str_col': np.random.choice(['foo', 'bar', 'baz', None], n_samples),
        'date_col': date_range,
        'missing_col': np.where(np.random.random(n_samples) > 0.7, np.nan, np.random.random(n_samples))
    }
    
    # Add some special values
    data['float_col'][0] = np.inf  # Add infinity
    data['float_col'][1] = -np.inf  # Add negative infinity
    data['float_col'][2] = np.nan  # Add NaN
    
    df = pd.DataFrame(data)
    # Add a non-trivial index
    df.index = pd.RangeIndex(start=1000, stop=1000 + n_samples, name='custom_index')
    return df

def test_basic_profile(sample_df, tmp_path):
    """Test basic profile generation"""
    output_file = tmp_path / "report.html"
    profile(sample_df, output_file=str(output_file))
    assert output_file.exists()

def test_pdf_export(sample_df, tmp_path):
    """Test PDF export functionality"""
    output_file = tmp_path / "report.pdf"
    profile(sample_df, output_file=str(output_file), output_format='pdf')
    assert output_file.exists()

def test_target_analysis(sample_df, tmp_path):
    """Test profiling with target variable"""
    output_file = tmp_path / "report.html"
    profile(sample_df, target='target', output_file=str(output_file))
    assert output_file.exists()

def test_data_size_limit():
    """Test data size limit enforcement"""
    # Create a DataFrame that exceeds the size limit
    big_df = pd.DataFrame(np.random.random((1_000_001, 5)))
    
    with pytest.raises(DataSizeError):
        profile(big_df, output_file="report.html")

def test_theme_options(sample_df, tmp_path):
    """Test theme customization"""
    output_file = tmp_path / "report.html"
    profile(sample_df, output_file=str(output_file), theme='dark')
    assert output_file.exists()
    
    # Verify theme is in the HTML
    content = output_file.read_text(encoding='utf-8')
    assert 'data-theme="dark"' in content

def test_compare_basic(sample_df, sample_df2):
    """Test basic DataFrame comparison functionality"""
    result = compare(sample_df, sample_df2)
    
    # Test columns only in first DataFrame
    assert set(result['columns_only_in_df1']) == {'target', 'missing'}
    
    # Test columns only in second DataFrame
    assert set(result['columns_only_in_df2']) == {'new_column', 'different_type'}
    
    # Test common columns
    assert set(result['common_columns']) == {'numeric', 'categorical'}
    
    # Test that numeric and categorical columns have same dtypes (no differences)
    assert 'numeric' not in result['dtype_differences']
    assert 'categorical' not in result['dtype_differences']

def test_compare_with_dtype_differences(sample_df):
    """Test DataFrame comparison with dtype differences"""
    # Create a modified version of sample_df with different dtypes
    df_modified = sample_df.copy()
    df_modified['numeric'] = df_modified['numeric'].astype('int64')  # Change float to int
    df_modified['categorical'] = df_modified['categorical'].astype('category')  # Change object to category
    
    result = compare(sample_df, df_modified)
    
    # Test dtype differences
    assert 'numeric' in result['dtype_differences']
    assert 'categorical' in result['dtype_differences']
    assert result['dtype_differences']['numeric'] == ('float64', 'int64')
    
    # Test that no columns are reported as unique to either DataFrame
    assert not result['columns_only_in_df1']
    assert not result['columns_only_in_df2']
    
    # Test that all columns are reported as common
    assert set(result['common_columns']) == set(sample_df.columns)

def test_compare_with_custom_names():
    """Test DataFrame comparison with custom DataFrame names"""
    df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df2 = pd.DataFrame({'b': [5, 6], 'c': [7, 8]})
    
    result = compare(df1, df2, name1="First DF", name2="Second DF")
    
    assert result['columns_only_in_df1'] == ['a']
    assert result['columns_only_in_df2'] == ['c']
    assert result['common_columns'] == ['b']
    assert not result['dtype_differences']  # No dtype differences for 'b' 

def test_compare_numeric_stats(sample_df, sample_df2):
    """Test statistical comparison of numeric columns"""
    result = compare(sample_df, sample_df2)
    
    # Check numeric column comparison
    numeric_stats = result['variable_comparison']['numeric']['stats']
    
    # Verify all required statistics are present
    assert set(numeric_stats.keys()) >= {
        'count', 'missing_count', 'missing_percent',
        'unique_count', 'unique_percent',
        'mean', 'std', 'min', 'q1', 'median', 'q3', 'max'
    }
    
    # Verify structure of numeric statistics
    for stat in ['mean', 'std', 'min', 'q1', 'median', 'q3', 'max']:
        assert 'df1' in numeric_stats[stat]
        assert 'df2' in numeric_stats[stat]
        # Verify values are formatted as strings with 2 decimal places
        assert '.' in numeric_stats[stat]['df1']
        assert len(numeric_stats[stat]['df1'].split('.')[-1]) == 2

def test_compare_categorical_stats(sample_df, sample_df2):
    """Test statistical comparison of categorical columns"""
    result = compare(sample_df, sample_df2)
    
    # Check categorical column comparison
    cat_stats = result['variable_comparison']['categorical']['stats']
    
    # Verify basic statistics are present
    assert set(cat_stats.keys()) >= {
        'count', 'missing_count', 'missing_percent',
        'unique_count', 'unique_percent',
        'top_values_df1', 'top_values_df2'
    }
    
    # Verify structure of top values
    for df_key in ['top_values_df1', 'top_values_df2']:
        assert len(cat_stats[df_key]) <= 5  # Should have at most 5 top values
        for value_info in cat_stats[df_key]:
            assert set(value_info.keys()) == {'value', 'count', 'percentage'}
            assert isinstance(value_info['value'], str)
            assert isinstance(value_info['count'], (int, np.int64))
            assert isinstance(value_info['percentage'], str)
            assert float(value_info['percentage'].rstrip('%')) <= 100

def test_compare_missing_values(sample_df):
    """Test comparison of columns with missing values"""
    # Create a modified version with different missing value patterns
    df_modified = sample_df.copy()
    df_modified.loc[0:10, 'missing'] = np.nan  # Different missing pattern
    
    result = compare(sample_df, df_modified)
    missing_stats = result['variable_comparison']['missing']['stats']
    
    # Verify missing value statistics
    assert missing_stats['missing_count']['df1'] != missing_stats['missing_count']['df2']
    assert float(missing_stats['missing_percent']['df1']) != float(missing_stats['missing_percent']['df2'])
    
    # Verify counts match the actual data
    assert missing_stats['missing_count']['df1'] == sample_df['missing'].isna().sum()
    assert missing_stats['missing_count']['df2'] == df_modified['missing'].isna().sum()

def test_compare_numeric_distribution(sample_df, sample_df2):
    """Test distribution data generation for numeric columns"""
    result = compare(sample_df, sample_df2)
    
    # Check numeric column distribution data
    dist_data = result['variable_comparison']['numeric']['distribution_data']
    
    # Verify structure
    assert dist_data['type'] == 'numeric'
    assert 'histogram' in dist_data
    assert 'kde' in dist_data
    
    # Check histogram data
    hist = dist_data['histogram']
    assert len(hist['bins']) == 31  # n_bins + 1 for edges
    assert len(hist['df1_counts']) == 30  # n_bins
    assert len(hist['df2_counts']) == 30
    assert all(isinstance(x, (int, float)) for x in hist['bins'])
    assert all(isinstance(x, (int, float)) for x in hist['df1_counts'])
    assert all(isinstance(x, (int, float)) for x in hist['df2_counts'])
    
    # Check KDE data
    kde = dist_data['kde']
    assert 'df1' in kde and 'df2' in kde
    for df_key in ['df1', 'df2']:
        assert 'x' in kde[df_key] and 'y' in kde[df_key]
        assert len(kde[df_key]['x']) == len(kde[df_key]['y'])
        assert len(kde[df_key]['x']) == 100  # Default points for KDE
        assert all(isinstance(x, (int, float)) for x in kde[df_key]['x'])
        assert all(isinstance(x, (int, float)) for x in kde[df_key]['y'])

def test_compare_categorical_distribution(sample_df, sample_df2):
    """Test distribution data generation for categorical columns"""
    result = compare(sample_df, sample_df2)
    
    # Check categorical column distribution data
    dist_data = result['variable_comparison']['categorical']['distribution_data']
    
    # Verify structure
    assert dist_data['type'] == 'categorical'
    assert 'value_counts' in dist_data
    assert 'df1' in dist_data['value_counts']
    assert 'df2' in dist_data['value_counts']
    
    # Check value counts
    vc1 = dist_data['value_counts']['df1']
    vc2 = dist_data['value_counts']['df2']
    
    # Verify df1 has expected categories
    assert set(vc1.keys()) == {'A', 'B', 'C'}
    assert sum(vc1.values()) == len(sample_df)  # Total counts should match DataFrame length
    
    # Verify df2 has expected categories
    assert set(vc2.keys()) == {'X', 'Y', 'Z'}
    assert sum(vc2.values()) == len(sample_df2)

def test_compare_with_custom_bins():
    """Test numeric distribution generation with custom bin count"""
    df1 = pd.DataFrame({'numeric': range(100)})
    df2 = pd.DataFrame({'numeric': range(50, 150)})
    
    result = compare(df1, df2, n_bins=20)
    dist_data = result['variable_comparison']['numeric']['distribution_data']
    
    # Verify custom bin count
    assert len(dist_data['histogram']['bins']) == 21  # n_bins + 1
    assert len(dist_data['histogram']['df1_counts']) == 20
    assert len(dist_data['histogram']['df2_counts']) == 20 

def test_compare_report_generation(tmp_path):
    """Test that the compare function generates an HTML report when output_file is specified."""
    # Create sample DataFrames
    df1 = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'C', 'B'],
        'only_df1': [1, 2, 3, 4, 5]
    })
    
    df2 = pd.DataFrame({
        'numeric': [2, 3, 4, 5, 6],
        'categorical': ['B', 'B', 'A', 'C', 'A'],
        'only_df2': [6, 7, 8, 9, 10]
    })
    
    # Generate report
    output_file = tmp_path / "comparison_report.html"
    results = compare(
        df1, 
        df2, 
        name1="First DF",
        name2="Second DF",
        output_file=str(output_file)
    )
    
    # Check that the report was generated
    assert output_file.exists()
    assert 'report_path' in results
    assert results['report_path'] == str(output_file)
    
    # Read the report content
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that key elements are present in the report
    assert "First DF" in content
    assert "Second DF" in content
    assert "numeric" in content
    assert "categorical" in content
    assert "only_df1" in content
    assert "only_df2" in content
    assert "Distribution Comparison" in content

def test_compare_report_themes(tmp_path):
    """Test that the compare function handles different themes correctly."""
    df1 = pd.DataFrame({'numeric': [1, 2, 3]})
    df2 = pd.DataFrame({'numeric': [2, 3, 4]})
    
    # Test light theme
    light_output = tmp_path / "light_theme.html"
    compare(df1, df2, output_file=str(light_output), theme="light")
    
    with open(light_output, 'r', encoding='utf-8') as f:
        light_content = f.read()
    
    assert 'data-theme="light"' in light_content
    
    # Test dark theme
    dark_output = tmp_path / "dark_theme.html"
    compare(df1, df2, output_file=str(dark_output), theme="dark")
    
    with open(dark_output, 'r', encoding='utf-8') as f:
        dark_content = f.read()
    
    assert 'data-theme="dark"' in dark_content

def test_compare_report_no_output():
    """Test that compare function works correctly when no output file is specified."""
    df1 = pd.DataFrame({'numeric': [1, 2, 3]})
    df2 = pd.DataFrame({'numeric': [2, 3, 4]})
    
    results = compare(df1, df2)
    assert 'report_path' not in results
    assert 'variable_comparison' in results
    assert 'numeric' in results['variable_comparison'] 

def test_profile_variables_structure(sample_df, tmp_path):
    """Test that variables are passed as a dictionary in the profile function"""
    output_file = tmp_path / "report.html"
    profile(sample_df, output_file=str(output_file))
    
    # Read the generated report
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that the report contains variable names from the sample DataFrame
    for col in sample_df.columns:
        if col != 'target':  # Skip target column if it exists
            assert col in content, f"Variable {col} not found in report"

def test_template_loading():
    """Test that templates can be loaded correctly"""
    env = Environment(loader=PackageLoader('pytics', 'templates'))
    env.globals['len'] = builtins.len
    
    # Test loading all templates
    templates = ['report_template.html.j2', 'compare_report_template.html.j2', 'base_template.html.j2']
    for template_name in templates:
        template = env.get_template(template_name)
        assert template is not None

def test_compare_report_context(sample_df):
    """Test that compare report has all required context variables"""
    from pytics.profiler import compare
    
    # Create a second DataFrame with some differences
    df2 = sample_df.copy()
    df2['new_column'] = range(len(df2))
    
    # Generate comparison report
    results = compare(sample_df, df2, output_file=None)
    
    # Check required context variables
    required_vars = [
        'columns_only_in_df1',
        'columns_only_in_df2',
        'common_columns',
        'dtype_differences',
        'variable_comparison',
        'df1',
        'df2'
    ]
    
    for var in required_vars:
        assert var in results, f"Missing required context variable: {var}"
    
    # Verify DataFrame references
    assert results['df1'] is sample_df
    assert results['df2'] is df2 

def test_json_export_basic(complex_df, tmp_path):
    """Test basic JSON export functionality"""
    output_file = tmp_path / "report.json"
    result_path = profile(complex_df, output_file=str(output_file), output_format='json')
    
    # Verify file was created
    assert Path(result_path).exists()
    assert result_path.endswith('.json')
    
    # Load and verify JSON structure
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check basic structure
    assert set(data.keys()) >= {'metadata', 'overview', 'variables', 'index_analysis'}
    
    # Check metadata
    assert data['metadata']['schema_version'] == '1.0'
    assert isinstance(data['metadata']['generated_at'], str)
    assert isinstance(data['metadata']['pytics_version'], str)
    
    # Check overview
    assert data['overview']['shape'] == {'rows': 100, 'columns': 7}
    assert isinstance(data['overview']['memory_usage'], str)
    assert isinstance(data['overview']['avg_record_size'], str)
    
    # Check index analysis
    assert data['index_analysis']['name'] == 'custom_index'
    assert data['index_analysis']['unique_count'] == 100
    assert isinstance(data['index_analysis']['memory_usage'], (int, float))

def test_json_export_variables(complex_df, tmp_path):
    """Test variable-specific details in JSON export"""
    output_file = tmp_path / "report.json"
    result_path = profile(complex_df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    variables = data['variables']
    
    # Test numeric column (float_col)
    float_var = variables['float_col']
    assert float_var['type'] == 'float64'
    assert set(float_var['statistics'].keys()) >= {
        'mean', 'std', 'min', 'max', 'median', 'q1', 'q3',
        'sum', 'skewness', 'kurtosis'
    }
    assert float_var['distribution']['type'] == 'histogram'
    assert 'memory_usage' in float_var
    
    # Test boolean column
    bool_var = variables['bool_col']
    assert bool_var['type'] == 'bool'
    assert set(bool_var['statistics'].keys()) >= {
        'true_count', 'false_count', 'true_percent', 'false_percent'
    }
    assert bool_var['distribution']['type'] == 'boolean'
    
    # Test datetime column
    date_var = variables['date_col']
    assert pd.api.types.is_datetime64_any_dtype(complex_df['date_col'].dtype)
    assert set(date_var['statistics'].keys()) >= {'min_date', 'max_date', 'range'}
    assert date_var['distribution']['type'] == 'datetime'
    
    # Test categorical column
    cat_var = variables['cat_col']
    assert set(cat_var['statistics'].keys()) >= {
        'distinct_count', 'top_frequent_value', 'frequency'
    }
    assert cat_var['distribution']['type'] == 'categorical'

def test_json_export_section_filtering(complex_df, tmp_path):
    """Test section filtering in JSON export"""
    output_file = tmp_path / "report.json"
    
    # Test including specific sections
    include_sections = ['metadata', 'overview', 'variables']
    result_path = profile(
        complex_df,
        output_file=str(output_file),
        output_format='json',
        include_sections=include_sections
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert set(data.keys()) == set(include_sections)
    
    # Test excluding specific sections
    exclude_sections = ['correlations', 'duplicates']
    result_path = profile(
        complex_df,
        output_file=str(output_file),
        output_format='json',
        exclude_sections=exclude_sections
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert not any(section in data for section in exclude_sections)
    assert 'metadata' in data  # metadata should always be present

def test_json_export_special_values(complex_df, tmp_path):
    """Test handling of special values in JSON export"""
    output_file = tmp_path / "report.json"
    result_path = profile(complex_df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check handling of special values in float column
    float_data = data['variables']['float_col']
    dist_data = float_data['distribution']
    
    # Verify special values are properly serialized
    assert isinstance(dist_data['counts'], list)
    assert all(isinstance(x, (int, float)) for x in dist_data['counts'])
    assert all(isinstance(x, (int, float)) for x in dist_data['bin_edges'])
    
    # Verify statistics with potential special values are properly handled
    stats = float_data['statistics']
    assert all(isinstance(v, (int, float, str)) for v in stats.values())

def test_json_export_target_analysis(complex_df, tmp_path):
    """Test JSON export with target variable analysis"""
    output_file = tmp_path / "report.json"
    result_path = profile(
        complex_df,
        target='bool_col',
        output_file=str(output_file),
        output_format='json'
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Verify target analysis is included
    assert 'target_analysis' in data
    target_data = data['target_analysis']
    
    # Check target variable statistics
    assert isinstance(target_data, dict)
    assert 'statistics' in target_data
    assert 'distribution' in target_data

def test_json_export_errors(complex_df, tmp_path):
    """Test error handling in JSON export"""
    # Test with invalid file path
    with pytest.raises(ProfilerError):
        profile(complex_df, output_file='/invalid/path/report.json', output_format='json')
    
    # Test with invalid section names
    with pytest.raises(ProfilerError):
        profile(
            complex_df,
            output_file=str(tmp_path / "report.json"),
            output_format='json',
            include_sections=['invalid_section']
        )

def test_json_export_memory_usage(complex_df, tmp_path):
    """Test memory usage reporting in JSON export"""
    output_file = tmp_path / "report.json"
    result_path = profile(complex_df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check overall memory usage
    assert 'memory_usage' in data['overview']
    assert 'avg_record_size' in data['overview']
    
    # Check per-column memory usage
    for col_name, col_data in data['variables'].items():
        assert 'memory_usage' in col_data
        assert isinstance(col_data['memory_usage'], int)
        assert col_data['memory_usage'] > 0

def test_json_export_datetime_handling(tmp_path):
    """Test handling of datetime data in JSON export"""
    # Create DataFrame with various datetime scenarios
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=5),
        'datetime_tz': pd.date_range('2023-01-01', periods=5, tz='UTC'),
        'datetime_with_null': pd.Series([pd.Timestamp('2023-01-01'), None, pd.Timestamp('2023-01-03')])
    })
    
    output_file = tmp_path / "report.json"
    result_path = profile(df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check datetime columns
    for col in ['datetime', 'datetime_tz', 'datetime_with_null']:
        var_data = data['variables'][col]
        assert var_data['type'].startswith('datetime64')
        if len(df[col].dropna()) > 0:
            stats = var_data['statistics']
            assert 'min_date' in stats
            assert 'max_date' in stats
            assert 'range' in stats
            # Verify dates are ISO format strings
            datetime.fromisoformat(stats['min_date'].replace('Z', '+00:00'))
            datetime.fromisoformat(stats['max_date'].replace('Z', '+00:00'))

def test_json_schema_consistency(complex_df, tmp_path):
    """Test consistency of JSON schema across multiple runs"""
    output_file1 = tmp_path / "report1.json"
    output_file2 = tmp_path / "report2.json"
    
    # Generate two reports with the same data
    result_path1 = profile(complex_df, output_file=str(output_file1), output_format='json')
    result_path2 = profile(complex_df, output_file=str(output_file2), output_format='json')
    
    with open(result_path1, 'r', encoding='utf-8') as f1, \
         open(result_path2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # Compare structure (ignoring values that should differ like timestamps)
    def compare_structure(d1, d2, path=""):
        assert isinstance(d1, type(d2)), f"Type mismatch at {path}"
        if isinstance(d1, dict):
            assert set(d1.keys()) == set(d2.keys()), f"Keys differ at {path}"
            for k in d1:
                if k != 'generated_at':  # Skip timestamp comparison
                    compare_structure(d1[k], d2[k], f"{path}.{k}")
        elif isinstance(d1, list):
            assert len(d1) == len(d2), f"List length differs at {path}"
            for i, (v1, v2) in enumerate(zip(d1, d2)):
                compare_structure(v1, v2, f"{path}[{i}]")
    
    compare_structure(data1, data2)

def test_json_schema_validation_basic(complex_df, tmp_path):
    """Test that JSON output validates against the schema for basic data types"""
    output_file = tmp_path / "report.json"
    result_path = profile(complex_df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)

def test_json_schema_validation_all_types(tmp_path):
    """Test schema validation with all possible data types"""
    # Create DataFrame with all possible types
    df = pd.DataFrame({
        'int': [1, 2, 3],
        'float': [1.1, np.inf, np.nan],
        'bool': [True, False, True],
        'category': pd.Categorical(['A', 'B', 'A']),
        'string': ['foo', None, 'bar'],
        'datetime': pd.date_range('2023-01-01', periods=3),
        'timedelta': pd.timedelta_range('1 day', periods=3),
        'complex': [1+2j, 3+4j, 5+6j]
    })
    
    output_file = tmp_path / "report.json"
    result_path = profile(df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)

def test_json_schema_validation_empty_df(tmp_path):
    """Test schema validation with an empty DataFrame"""
    df = pd.DataFrame()
    output_file = tmp_path / "report.json"
    result_path = profile(df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)
    
    # Check specific requirements for empty DataFrame
    assert data['overview']['shape']['rows'] == 0
    assert data['overview']['shape']['columns'] == 0
    assert data['overview']['n_vars'] == 0
    assert not data['variables']  # Should be empty dict

def test_json_schema_validation_section_filtering(complex_df, tmp_path):
    """Test schema validation with section filtering"""
    output_file = tmp_path / "report.json"
    
    # Test with minimal required sections
    result_path = profile(
        complex_df,
        output_file=str(output_file),
        output_format='json',
        include_sections=['metadata', 'overview', 'variables']
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)
    
    # Test excluding optional sections
    result_path = profile(
        complex_df,
        output_file=str(output_file),
        output_format='json',
        exclude_sections=['correlations', 'duplicates', 'missing_values']
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)

def test_json_schema_validation_target_analysis(complex_df, tmp_path):
    """Test schema validation with target analysis"""
    output_file = tmp_path / "report.json"
    result_path = profile(
        complex_df,
        target='bool_col',
        output_file=str(output_file),
        output_format='json'
    )
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)
    
    # Verify target analysis structure
    assert 'target_analysis' in data
    assert set(data['target_analysis'].keys()) >= {'statistics', 'distribution'}

def test_json_schema_validation_special_values(tmp_path):
    """Test schema validation with special numeric values"""
    df = pd.DataFrame({
        'special_float': [np.inf, -np.inf, np.nan, 1.0],
        'zero_div': [1.0/0.0, -1.0/0.0, 0.0/0.0, 1.0],
        'large_numbers': [1e308, -1e308, 1e-308, 0]
    })
    
    output_file = tmp_path / "report.json"
    result_path = profile(df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)
    
    # Check special value handling
    for col in ['special_float', 'zero_div']:
        var_data = data['variables'][col]
        stats = var_data['statistics']
        # Verify infinity and NaN are properly serialized as strings
        assert isinstance(stats['max'], (str, float))
        assert isinstance(stats['min'], (str, float))
        assert isinstance(stats['mean'], (str, float))

def test_json_schema_validation_datetime_formats(tmp_path):
    """Test schema validation with various datetime formats"""
    df = pd.DataFrame({
        'datetime_naive': pd.date_range('2023-01-01', periods=3),
        'datetime_tz': pd.date_range('2023-01-01', periods=3, tz='UTC'),
        'datetime_mixed': [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-02', tz='US/Pacific'),
            None
        ]
    })
    
    output_file = tmp_path / "report.json"
    result_path = profile(df, output_file=str(output_file), output_format='json')
    
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Should not raise ValidationError
    validate(instance=data, schema=PYTICS_JSON_SCHEMA)
    
    # Check datetime format compliance
    for col in data['variables']:
        if data['variables'][col]['type'].startswith('datetime'):
            stats = data['variables'][col]['statistics']
            # Verify dates are valid ISO format
            if 'min_date' in stats:
                datetime.fromisoformat(stats['min_date'].replace('Z', '+00:00'))
            if 'max_date' in stats:
                datetime.fromisoformat(stats['max_date'].replace('Z', '+00:00'))

def test_json_schema_validation_invalid_data():
    """Test that invalid data structures are caught by schema validation"""
    # Test case 1: Missing required field
    invalid_data = {
        "metadata": {
            "title": "Test Report",
            # Missing required field: generated_at
            "pytics_version": "1.0.0",
            "schema_version": "1.0"
        },
        "overview": {},
        "variables": {}
    }
    
    with pytest.raises(ValidationError):
        validate(instance=invalid_data, schema=PYTICS_JSON_SCHEMA)
    
    # Test case 2: Invalid numeric value
    invalid_data = {
        "metadata": {
            "title": "Test Report",
            "generated_at": "2023-01-01T00:00:00",
            "pytics_version": "1.0.0",
            "schema_version": "1.0"
        },
        "overview": {
            "shape": {"rows": -1, "columns": 1}  # Invalid negative value
        },
        "variables": {}
    }
    
    with pytest.raises(ValidationError):
        validate(instance=invalid_data, schema=PYTICS_JSON_SCHEMA)
    
    # Test case 3: Invalid percentage value
    invalid_data = {
        "metadata": {
            "title": "Test Report",
            "generated_at": "2023-01-01T00:00:00",
            "pytics_version": "1.0.0",
            "schema_version": "1.0"
        },
        "overview": {
            "shape": {"rows": 10, "columns": 1},
            "missing_percent": 150.0  # Invalid percentage > 100
        },
        "variables": {}
    }
    
    with pytest.raises(ValidationError):
        validate(instance=invalid_data, schema=PYTICS_JSON_SCHEMA) 