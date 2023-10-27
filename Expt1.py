# Gradient descent : https://colab.research.google.com/drive/1irODBfH1KXcy9XrYkVhsgX3vbuwXoRQ0#scrollTo=BuflF8sGNQXF

# Least Square:
# https://colab.research.google.com/drive/1tV9g-qkpGAWQTRL68XjUBj6USwet-pRf


#ml preprocessing imputation,anomaly detection, standardization, normalization, encoding
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Create the DataFrame

data = {
'Student ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'Exam 1 Score': [85, 76, 90, 65, 88, None, 78, 92, 85, 70],
'Exam 2 Score': [92, 78, 88, 75, 91, 82, 76, 96, 89, 68],
'Exam 3 Score': [88, None, 94, 80, 87, 79, 72, 98, 91, 75],
'Final Grade': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'A', 'B']
}
df = pd.DataFrame(data)

# Imputation
imputer = SimpleImputer(strategy="mean")
df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']] = imputer.fit_transform(df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']])

# Anomaly Detection
z_scores = stats.zscore(df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']])
threshold = 3
outliers = (abs(z_scores) > threshold).any(axis=1)
df['Is Outlier'] = outliers

# Standardization
scaler = StandardScaler()
df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']] = scaler.fit_transform(df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']])

# Normalization
scaler = MinMaxScaler()
df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']] = scaler.fit_transform(df[['Exam 1 Score', 'Exam 2 Score', 'Exam 3 Score']])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Final Grade'], prefix=['Grade'])


# Output
print("Final Preprocessed Dataset:\n", df.to_markdown())
