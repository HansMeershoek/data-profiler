"""
Core profiling functionality
"""
from typing import Optional, List, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import jinja2
from xhtml2pdf import pisa

class ProfilerError(Exception):
    """Base exception for data profiler errors"""
    pass

class DataSizeError(ProfilerError):
    """Exception raised when data size exceeds limits"""
    pass

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

    # Map theme to Plotly template
    plotly_template = 'plotly_white' if theme == 'light' else 'plotly_dark'

    # Create the main figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Types', 'Missing Values', 'Target Distribution', 'Correlations')
    )
    
    # Data Types plot
    type_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Bar(x=type_counts.index.astype(str), y=type_counts.values, name='Data Types'),
        row=1, col=1
    )
    
    # Missing Values plot
    missing = df.isnull().sum()
    fig.add_trace(
        go.Bar(x=missing.index, y=missing.values, name='Missing Values'),
        row=1, col=2
    )
    
    # Target Distribution plot (if target is specified)
    if target and target in df.columns:
        if df[target].dtype in ['int64', 'float64']:
            fig.add_trace(
                go.Histogram(x=df[target], name='Target Distribution'),
                row=2, col=1
            )
        else:
            target_counts = df[target].value_counts()
            fig.add_trace(
                go.Bar(x=target_counts.index.astype(str), y=target_counts.values, name='Target Distribution'),
                row=2, col=1
            )
    
    # Correlations plot (for numeric columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr, x=corr.columns, y=corr.columns, name='Correlations'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=False,
        template=plotly_template
    )
    
    # Create HTML report
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ title }}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: {{ 'white' if theme == 'light' else '#1a1a1a' }};
                color: {{ 'black' if theme == 'light' else 'white' }};
            }
            .overview {
                background-color: {{ '#f5f5f5' if theme == 'light' else '#2d2d2d' }};
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>{{ title }}</h1>
        <div class="overview">
            <h2>Dataset Overview</h2>
            <p>Rows: {{ rows }}</p>
            <p>Columns: {{ cols }}</p>
            <p>Memory Usage: {{ memory_usage }}</p>
        </div>
        {{ plot_div }}
    </body>
    </html>
    """
    
    context = {
        'title': title,
        'theme': theme,
        'rows': len(df),
        'cols': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'plot_div': fig.to_html(full_html=False)
    }
    
    html_report = jinja2.Template(template).render(**context)
    
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