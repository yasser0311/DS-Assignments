# Employee Performance and Retention Analysis

## Overview
This Jupyter Notebook analyzes employee performance and retention data to identify patterns, trends, and factors influencing employee attrition. The analysis includes data collection, preprocessing, exploratory data analysis (EDA), and predictive modeling using TensorFlow.

## Dataset
The dataset contains the following columns:
- `EmployeeID`: Unique identifier for each employee
- `Name`: Employee name
- `Age`: Employee age
- `Department`: Department the employee belongs to
- `Salary`: Employee salary
- `YearsAtCompany`: Number of years the employee has been with the company
- `PerformanceScore`: Employee performance score
- `Attrition`: Whether the employee has left the company (Yes/No)

## Notebook Structure
1. **Data Collection and Preprocessing**
   - Load the dataset
   - Check for missing values and duplicates
   - Basic data information

2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics for numerical columns
   - Visualization of data distributions and relationships

3. **Predictive Modeling**
   - Data preprocessing (label encoding, standardization)
   - Train-test split
   - Build and evaluate a neural network model using TensorFlow/Keras

## Key Findings
- The dataset contains 100 entries with no missing values or duplicates.
- Employees have an average age of 36.57 years and an average salary of $72,820.
- The average performance score is 84.94, with a standard deviation of 6.35.
- The dataset includes both numerical (age, salary, years at company, performance score) and categorical (department, attrition) features.

## Dependencies
- Python 3.10
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scipy
- scikit-learn

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Jupyter Notebook to perform the analysis.
3. The notebook includes code for data preprocessing, visualization, and model training.

## Results
The analysis provides insights into employee performance and retention patterns, which can help HR departments develop strategies to improve employee satisfaction and reduce attrition.

## License
This project is open-source and available for anyone to use or modify.