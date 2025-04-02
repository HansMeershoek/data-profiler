"""
Tests for the data profiler functionality
"""
import pytest
import pandas as pd
import numpy as np
from pytics import profile
from pytics.profiler import DataSizeError, ProfilerError
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