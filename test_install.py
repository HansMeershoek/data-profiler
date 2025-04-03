import pandas as pd
import numpy as np
from pandas_profiler import profile

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'education_years': np.random.randint(8, 22, n_samples),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'last_purchase': pd.date_range('2023-01-01', periods=n_samples).to_numpy()
}

df = pd.DataFrame(data)

# Generate HTML report
print("Generating HTML report...")
profile(
    df,
    target='churn',
    output_file='test_report.html',
    title='Customer Data Profile'
)

# Generate PDF report
print("Generating PDF report...")
profile(
    df,
    target='churn',
    output_file='test_report.pdf',
    output_format='pdf',
    title='Customer Data Profile'
) 