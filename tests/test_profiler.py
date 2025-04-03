"""
Tests for the data profiler functionality
"""
import pytest
import pandas as pd
import numpy as np
from pytics import profile
from pytics.profiler import DataSizeError, ProfilerError, compare
from pathlib import Path

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
    assert 'background-color: #1a1a1a' in content 

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