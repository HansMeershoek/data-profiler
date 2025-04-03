"""
Visualization functions for data profiling and comparison
"""
from typing import Dict, Any, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
from io import BytesIO

def _convert_to_static_image(fig: go.Figure, format: str = 'png') -> str:
    """
    Convert a Plotly figure to a static image and return as base64 string.
    
    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to convert
    format : str, default 'png'
        The image format to use
        
    Returns
    -------
    str
        Base64 encoded image string with data URI prefix
    """
    img_bytes = pio.to_image(fig, format=format, engine='kaleido')
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/{format};base64,{base64_image}"

def create_distribution_comparison_plot(
    distribution_data: Dict[str, Any],
    name1: str = "DataFrame 1",
    name2: str = "DataFrame 2",
    theme: str = "light",
    return_static: bool = False
) -> Union[str, Tuple[str, go.Figure]]:
    """
    Create a comparison plot for distributions using Plotly.
    
    Parameters
    ----------
    distribution_data : Dict[str, Any]
        Distribution data for both DataFrames
    name1 : str, default "DataFrame 1"
        Name of the first DataFrame
    name2 : str, default "DataFrame 2"
        Name of the second DataFrame
    theme : str, default "light"
        Theme for the plot ('light' or 'dark')
    return_static : bool, default False
        If True, return a base64 encoded static image instead of interactive HTML
        
    Returns
    -------
    Union[str, Tuple[str, go.Figure]]
        If return_static is True: base64 encoded image string
        If return_static is False: tuple of (HTML string, Figure object)
    """
    if distribution_data['type'] == 'numeric':
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add histogram traces
        fig.add_trace(go.Histogram(
            x=distribution_data['histogram']['bins'][:-1],
            y=distribution_data['histogram']['df1_counts'],
            name=name1,
            marker_color='#2ecc71',  # Green
            opacity=0.75
        ))
        
        fig.add_trace(go.Histogram(
            x=distribution_data['histogram']['bins'][:-1],
            y=distribution_data['histogram']['df2_counts'],
            name=name2,
            marker_color='#f1c40f',  # Yellow
            opacity=0.75
        ))
        
        # Add KDE traces
        fig.add_trace(go.Scatter(
            x=distribution_data['kde']['df1']['x'],
            y=distribution_data['kde']['df1']['y'],
            name=f"{name1} KDE",
            line=dict(color='#2ecc71', width=2),
            yaxis='y2'
        ))
        
        fig.add_trace(go.Scatter(
            x=distribution_data['kde']['df2']['x'],
            y=distribution_data['kde']['df2']['y'],
            name=f"{name2} KDE",
            line=dict(color='#f1c40f', width=2),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Distribution Comparison',
            xaxis_title='Value',
            yaxis_title='Count',
            yaxis2=dict(
                title='Density',
                overlaying='y',
                side='right'
            ),
            template='plotly_white' if theme == 'light' else 'plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if return_static:
            return _convert_to_static_image(fig)
        else:
            return fig.to_html(full_html=False, include_plotlyjs='cdn'), fig
    else:
        # For categorical data
        # Get all unique categories from both DataFrames
        categories = set()
        categories.update(distribution_data['value_counts']['df1'].keys())
        categories.update(distribution_data['value_counts']['df2'].keys())
        categories = sorted(list(categories))
        
        # Create traces for each DataFrame
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=[distribution_data['value_counts']['df1'].get(cat, 0) for cat in categories],
            name=name1,
            marker_color='#2ecc71'  # Green
        ))
        
        fig.add_trace(go.Bar(
            x=categories,
            y=[distribution_data['value_counts']['df2'].get(cat, 0) for cat in categories],
            name=name2,
            marker_color='#f1c40f'  # Yellow
        ))
        
        # Update layout
        fig.update_layout(
            title='Category Distribution Comparison',
            xaxis_title='Category',
            yaxis_title='Count',
            template='plotly_white' if theme == 'light' else 'plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            barmode='group'
        )
        
        if return_static:
            return _convert_to_static_image(fig)
        else:
            return fig.to_html(full_html=False, include_plotlyjs='cdn'), fig 