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