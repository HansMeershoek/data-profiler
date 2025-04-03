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