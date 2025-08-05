# Python Data Analysis Project 📊

## 📌 Project Overview
This project demonstrates essential data analysis techniques using Python, covering:
- Data loading and cleaning
- Exploratory analysis
- Visualization with Matplotlib/Seaborn
- Basic statistical operations

## 🛠️ Tasks

### Task 1: Data Loading & Cleaning
```python
import pandas as pd

# Load dataset
try:
    df = pd.read_csv('your_dataset.csv')  # Replace with your file
    print(df.head())  # Inspect first 5 rows
    print(df.info())  # Check structure and missing values
    
    # Handle missing data
    df = df.dropna()  # or df.fillna(value)
except FileNotFoundError:
    print("Error: File not found!")
except Exception as e:
    print(f"An error occurred: {str(e)}")
