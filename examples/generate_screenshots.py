"""
Script to generate example reports and screenshots for documentation.
"""
import pandas as pd
import numpy as np
from pandas_profiler import profile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'education_years': np.random.randint(8, 22, n_samples),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'last_purchase': pd.date_range(start='2023-01-01', periods=n_samples),
    'missing_column': np.where(np.random.random(n_samples) > 0.7, np.nan, np.random.random(n_samples))
}

df = pd.DataFrame(data)

# Generate full report
profile(
    df,
    target='churn',
    output_file='examples/full_report.html',
    title='Complete Data Profile Report'
)

# Generate targeted report with specific sections
profile(
    df,
    target='churn',
    include_sections=['overview', 'target_analysis'],
    output_file='examples/targeted_report.html',
    title='Targeted Analysis Report',
    theme='dark'
)

# Generate PDF report
profile(
    df,
    target='churn',
    output_format='pdf',
    output_file='examples/report.pdf',
    title='PDF Profile Report'
)

# Take screenshots using Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=chrome_options)

# Screenshot full report
driver.get('file://' + os.path.abspath('examples/full_report.html'))
time.sleep(2)  # Wait for animations
driver.save_screenshot('examples/full_report.png')

# Screenshot targeted report
driver.get('file://' + os.path.abspath('examples/targeted_report.html'))
time.sleep(2)  # Wait for animations
driver.save_screenshot('examples/targeted_report.png')

driver.quit()

print("Reports and screenshots generated successfully!") 