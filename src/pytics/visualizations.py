"""
Visualization functions for data profiling and comparison
"""
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_distribution_comparison_plot(
    distribution_data: Dict[str, Any],
    name1: str = "DataFrame 1",
    name2: str = "DataFrame 2",
    theme: str = "light"
) -> str:
    """
    Create a comparison plot for distributions between two DataFrames.

    Parameters
    ----------
    distribution_data : Dict[str, Any]
        Distribution data from compare() function
    name1 : str
        Name of the first DataFrame
    name2 : str
        Name of the second DataFrame
    theme : str
        'light' or 'dark' theme

    Returns
    -------
    str
        HTML representation of the plot
    """
    plotly_template = 'plotly_white' if theme == 'light' else 'plotly_dark'
    
    if distribution_data['type'] == 'numeric':
        # Create figure with secondary y-axis for KDE
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add histogram for df1
        fig.add_trace(
            go.Histogram(
                x=distribution_data['histogram']['bins'][:-1],
                y=distribution_data['histogram']['df1_counts'],
                name=f"{name1} Histogram",
                opacity=0.7,
                nbinsx=len(distribution_data['histogram']['bins']) - 1
            ),
            secondary_y=False
        )
        
        # Add histogram for df2
        fig.add_trace(
            go.Histogram(
                x=distribution_data['histogram']['bins'][:-1],
                y=distribution_data['histogram']['df2_counts'],
                name=f"{name2} Histogram",
                opacity=0.7,
                nbinsx=len(distribution_data['histogram']['bins']) - 1
            ),
            secondary_y=False
        )
        
        # Add KDE for df1
        fig.add_trace(
            go.Scatter(
                x=distribution_data['kde']['df1']['x'],
                y=distribution_data['kde']['df1']['y'],
                name=f"{name1} Density",
                line=dict(dash='dash')
            ),
            secondary_y=True
        )
        
        # Add KDE for df2
        fig.add_trace(
            go.Scatter(
                x=distribution_data['kde']['df2']['x'],
                y=distribution_data['kde']['df2']['y'],
                name=f"{name2} Density",
                line=dict(dash='dash')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Distribution Comparison",
            barmode='overlay',
            template=plotly_template
        )
        
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Density", secondary_y=True)
        
    else:  # categorical
        # Get all unique categories
        all_categories = set(distribution_data['value_counts']['df1'].keys()) | \
                        set(distribution_data['value_counts']['df2'].keys())
        
        # Create traces for both DataFrames
        fig = go.Figure()
        
        # Add bars for df1
        fig.add_trace(go.Bar(
            name=name1,
            x=list(all_categories),
            y=[distribution_data['value_counts']['df1'].get(cat, 0) for cat in all_categories],
            opacity=0.7
        ))
        
        # Add bars for df2
        fig.add_trace(go.Bar(
            name=name2,
            x=list(all_categories),
            y=[distribution_data['value_counts']['df2'].get(cat, 0) for cat in all_categories],
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Category Distribution Comparison",
            barmode='group',
            template=plotly_template,
            xaxis_title="Categories",
            yaxis_title="Count"
        )
    
    return fig.to_html(full_html=False) 