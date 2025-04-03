"""
Core profiling functionality
"""
from typing import Optional, List, Literal, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import jinja2
from xhtml2pdf import pisa
import os

class ProfilerError(Exception):
    """Base exception for data profiler errors"""
    pass

class DataSizeError(ProfilerError):
    """Exception raised when data size exceeds limits"""
    pass

def compare(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "DataFrame 1",
    name2: str = "DataFrame 2",
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two pandas DataFrames and analyze their differences, starting with schema comparison.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame to compare
    df2 : pandas.DataFrame
        Second DataFrame to compare
    name1 : str, default "DataFrame 1"
        Name to identify the first DataFrame in the comparison
    name2 : str, default "DataFrame 2"
        Name to identify the second DataFrame in the comparison
    output_file : str, optional
        Path to save the comparison report (not implemented yet)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing schema comparison results:
        - columns_only_in_df1: List of columns present only in df1
        - columns_only_in_df2: List of columns present only in df2
        - common_columns: List of columns present in both DataFrames
        - dtype_differences: Dict mapping column names to tuple of (df1_dtype, df2_dtype)
          for columns with different dtypes
    """
    # Get column sets
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    # Find unique and common columns
    columns_only_in_df1 = sorted(list(cols1 - cols2))
    columns_only_in_df2 = sorted(list(cols2 - cols1))
    common_columns = sorted(list(cols1 & cols2))

    # Analyze dtype differences for common columns
    dtype_differences = {}
    for col in common_columns:
        dtype1 = str(df1[col].dtype)
        dtype2 = str(df2[col].dtype)
        if dtype1 != dtype2:
            dtype_differences[col] = (dtype1, dtype2)

    return {
        'columns_only_in_df1': columns_only_in_df1,
        'columns_only_in_df2': columns_only_in_df2,
        'common_columns': common_columns,
        'dtype_differences': dtype_differences
    }

def _calculate_overview_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overview statistics for the DataFrame"""
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'duplicate_rows': duplicate_rows,
        'duplicate_rows_pct': f"{(duplicate_rows / len(df)) * 100:.2f}",
        'missing_cells': missing_cells,
        'missing_cells_pct': f"{(missing_cells / total_cells) * 100:.2f}",
        'avg_record_size': f"{df.memory_usage(deep=True).sum() / len(df) / 1024:.2f} KB"
    }

def _analyze_variable(df: pd.DataFrame, column: str, target: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a single variable/column"""
    series = df[column]
    total_count = len(series)
    missing_count = series.isna().sum()
    distinct_count = series.nunique()
    
    var_stats = {
        'name': column,
        'type': str(series.dtype),
        'distinct_count': distinct_count,
        'distinct_pct': f"{(distinct_count / total_count) * 100:.2f}",
        'missing_count': missing_count,
        'missing_pct': f"{(missing_count / total_count) * 100:.2f}"
    }
    
    # Add numeric statistics if applicable
    if series.dtype in ['int64', 'float64']:
        desc = series.describe()
        var_stats.update({
            'mean': f"{desc['mean']:.2f}",
            'std': f"{desc['std']:.2f}",
            'min': f"{desc['min']:.2f}",
            'q1': f"{desc['25%']:.2f}",
            'median': f"{desc['50%']:.2f}",
            'q3': f"{desc['75%']:.2f}",
            'max': f"{desc['max']:.2f}"
        })
        
        # Distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=series.dropna(), name=column))
        fig.add_trace(go.Box(x=series.dropna(), name=column, yaxis='y2'))
        fig.update_layout(
            title=f"{column} Distribution",
            yaxis2=dict(overlaying='y', side='right')
        )
        var_stats['distribution_plot'] = fig.to_html(full_html=False)
        
        # Target relationship plot if target exists
        if target and target in df.columns:
            if df[target].dtype in ['int64', 'float64']:
                fig = px.scatter(df, x=column, y=target, title=f"{column} vs {target}")
            else:
                fig = px.box(df, x=column, y=target, title=f"{column} by {target}")
            var_stats['target_plot'] = fig.to_html(full_html=False)
    else:
        # For categorical variables
        value_counts = series.value_counts()
        var_stats['mode'] = value_counts.index[0] if not value_counts.empty else None
        
        # Distribution plot
        fig = px.bar(
            x=value_counts.index[:20],  # Show top 20 categories
            y=value_counts.values[:20],
            title=f"{column} Distribution (Top 20 Categories)"
        )
        var_stats['distribution_plot'] = fig.to_html(full_html=False)
        
        # Target relationship plot if target exists
        if target and target in df.columns and df[target].dtype in ['int64', 'float64']:
            fig = px.box(df, x=column, y=target, title=f"{target} by {column}")
            var_stats['target_plot'] = fig.to_html(full_html=False)
    
    return var_stats

def _create_summary_plots(df: pd.DataFrame, target: Optional[str] = None, theme: str = 'light') -> Dict[str, str]:
    """Create summary plots for the report"""
    plotly_template = 'plotly_white' if theme == 'light' else 'plotly_dark'
    
    # Data Types and Missing Values plot
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Data Types', 'Missing Values'),
        column_widths=[0.5, 0.5]
    )
    
    # Data Types
    type_counts = df.dtypes.value_counts()
    fig1.add_trace(
        go.Bar(x=type_counts.index.astype(str), y=type_counts.values, name='Data Types'),
        row=1, col=1
    )
    
    # Missing Values
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    fig1.add_trace(
        go.Bar(
            x=missing.values,
            y=missing.index,
            orientation='h',
            name='Missing Values (%)'
        ),
        row=1, col=2
    )
    
    fig1.update_layout(
        height=400,
        showlegend=False,
        template=plotly_template
    )
    
    # Correlations plot
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig2 = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu'
        ))
        fig2.update_layout(
            title='Correlation Matrix',
            height=600,
            template=plotly_template
        )
        correlations_plot = fig2.to_html(full_html=False)
    else:
        correlations_plot = "<p>No numeric columns available for correlation analysis.</p>"
    
    # Target distribution plot
    if target and target in df.columns:
        if df[target].dtype in ['int64', 'float64']:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=df[target], name='Distribution'))
            fig3.add_trace(go.Box(x=df[target], name='Box Plot', yaxis='y2'))
            fig3.update_layout(
                title=f'{target} Distribution',
                yaxis2=dict(overlaying='y', side='right'),
                template=plotly_template
            )
        else:
            counts = df[target].value_counts()
            fig3 = px.bar(
                x=counts.index,
                y=counts.values,
                title=f'{target} Distribution'
            )
            fig3.update_layout(template=plotly_template)
        target_plot = fig3.to_html(full_html=False)
    else:
        target_plot = ""
    
    return {
        'types_and_missing': fig1.to_html(full_html=False),
        'correlations': correlations_plot,
        'target_distribution': target_plot
    }

def _analyze_duplicates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze duplicate rows in the DataFrame"""
    if df.duplicated().sum() == 0:
        return []
    
    dup_counts = df.groupby(list(df.columns)).size()
    dup_counts = dup_counts[dup_counts > 1].sort_values(ascending=False)
    
    return [
        {'count': count, 'rows': ', '.join(map(str, df[df.duplicated(keep=False)].index[:5]))}
        for count in dup_counts[:10]  # Show top 10 duplicate patterns
    ]

def profile(
    df: pd.DataFrame,
    target: Optional[str] = None,
    output_file: str = 'report.html',
    output_format: Literal['html', 'pdf'] = 'html',
    include_sections: Optional[List[str]] = None,
    exclude_sections: Optional[List[str]] = None,
    theme: Literal['light', 'dark'] = 'light',
    title: str = "Data Profile Report"
) -> None:
    """
    Generate a profile report for the given DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to profile
    target : str, optional
        Name of the target variable for supervised learning tasks
    output_file : str, default 'report.html'
        Path to save the report
    output_format : {'html', 'pdf'}, default 'html'
        Output format for the report
    include_sections : list of str, optional
        Sections to include in the report
    exclude_sections : list of str, optional
        Sections to exclude from the report
    theme : {'light', 'dark'}, default 'light'
        Color theme for the report
    title : str, default "Data Profile Report"
        Title for the report
        
    Raises
    ------
    DataSizeError
        If the DataFrame exceeds size limits
    ProfilerError
        For other profiling-related errors
    """
    # Check data size limits
    if len(df) > 1_000_000:
        raise DataSizeError("DataFrame exceeds 1 million rows limit")
    if len(df.columns) > 1000:
        raise DataSizeError("DataFrame exceeds 1000 columns limit")
    
    # Calculate all statistics and generate plots
    overview = _calculate_overview_stats(df)
    variables = [_analyze_variable(df, col, target) for col in df.columns if col != target]
    plots = _create_summary_plots(df, target, theme)
    duplicates = _analyze_duplicates(df)
    
    # Target variable analysis if specified
    target_analysis = None
    if target and target in df.columns:
        target_analysis = _analyze_variable(df, target)
    
    # Prepare template context
    context = {
        'title': title,
        'theme': theme,
        'overview': overview,
        'variables': variables,
        'plots': plots,
        'duplicates': duplicates,
        'target': target_analysis
    }
    
    # Load and render template
    template_path = Path(__file__).parent / 'templates' / 'report_template.html.j2'
    if not template_path.exists():
        raise ProfilerError(f"Template file not found at {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = jinja2.Template(f.read())
    
    html_report = template.render(**context)
    
    # Save the report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'html':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
    else:  # pdf
        pdf_path = output_path.with_suffix('.pdf')
        result_file = open(pdf_path, "w+b")
        pisa_status = pisa.CreatePDF(html_report, dest=result_file)
        result_file.close()
        
        if pisa_status.err:
            raise ProfilerError("Error generating PDF report")