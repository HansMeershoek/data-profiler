from pytics.visualizations import _convert_to_static_image
import plotly.graph_objects as go

# Create a simple figure
fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
fig.update_layout(title="Test Figure")

# Try to convert to static image
try:
    static_image = _convert_to_static_image(fig)
    print("Success! Function imported and working.")
    print(f"Static image type: {type(static_image)}")
    print(f"Static image starts with: {static_image[:50]}...")
except Exception as e:
    print(f"Error: {str(e)}") 