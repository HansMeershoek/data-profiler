import pandas as pd
import pytics # Make sure pytics v1.1.4 is installed
import os
import sys
import time
import traceback

print(f"--- Running with Python: {sys.version} ---")
print(f"--- Pytics version: {pytics.__version__} ---") # Verify version
print("Creating DataFrame...")
sys.stdout.flush()
df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
output_file = "standalone_test.pdf"
output_path = os.path.abspath(output_file)

print(f"Calling pytics.profile for PDF to {output_path}...")
sys.stdout.flush()
start_time = time.time()
try:
    pytics.profile(df, output_format='pdf', output_file=output_file)
    end_time = time.time()
    print(f"Profile function completed successfully in {end_time - start_time:.2f} seconds.")
    print(f"Output file exists: {os.path.exists(output_path)}")
    sys.stdout.flush()
except Exception as e:
    end_time = time.time()
    print(f"Profile function failed after {end_time - start_time:.2f} seconds: {e}")
    sys.stdout.flush()
    traceback.print_exc() # Print full traceback if it fails

print("Script finished.")
sys.stdout.flush() 