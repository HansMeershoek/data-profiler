import plotly.graph_objects as go
import time

print("Creating simple figure...")
fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[2, 1, 3]))

print("Attempting to write image using Kaleido...")
start_time = time.time()
try:
    fig.write_image("test_kaleido.png", engine='kaleido')
    end_time = time.time()
    print(f"Successfully created test_kaleido.png in {end_time - start_time:.2f} seconds.")
except Exception as e:
    end_time = time.time()
    print(f"Error during write_image after {end_time - start_time:.2f} seconds: {e}") 