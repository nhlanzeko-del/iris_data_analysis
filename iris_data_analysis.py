import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset using sklearn
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Task 1: Load and Explore the Dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset (if there were missing values)
df = df.dropna()

# Task 2: Basic Data Analysis
print("\nDescriptive statistics:")
print(df.describe())

print("\nMean of each numerical column grouped by species:")
grouped_means = df.groupby("target").mean()
print(grouped_means)

# Rename target numbers to species names for clarity in visualization
df['species'] = df['target'].apply(lambda i: iris.target_names[i])

# Task 3: Data Visualization

# 1. Line Chart - Simulated time trend using index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title("Sepal and Petal Length Trend Over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram - Sepal width distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs. Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.show()
