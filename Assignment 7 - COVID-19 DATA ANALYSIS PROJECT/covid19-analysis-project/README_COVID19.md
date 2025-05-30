# 🦠 COVID-19 Data Analysis Project

## 📌 Assignment Overview
  
**Assignment:** COVID-19 Data Analysis  
**Goal:** To analyze the global impact of COVID-19 in correlation with worldwide happiness indicators.

---

## 🧠 Problem Statement

This project explores the relationship between global happiness and the impact of COVID-19 using real-world datasets. We aim to uncover insights about how socio-economic well-being may have influenced or aligned with the effects of the pandemic.

---

## 📁 Datasets Used

1. **COVID-19 Dataset**  
   - Daily confirmed cases and deaths by country  
   - Source: Johns Hopkins University (as provided)

2. **Worldwide Happiness Report**  
   - Includes: Happiness Score, GDP per Capita, Life Expectancy, Social Support, Freedom, Generosity, and Corruption perception  
   - Source: World Happiness Report

---

## 🔍 Project Workflow

### ✅ Step 1: Data Loading & Initial Inspection
- Loaded all three datasets (confirmed cases, deaths, and happiness report)
- Checked for nulls, shapes, and initial structure

### ✅ Step 2: Data Cleaning & Preprocessing
- Removed unnecessary columns (Lat, Long, Province/State)
- Converted wide format to long format (melted COVID datasets)
- Cleaned happiness column names for consistency
- Merged COVID and happiness datasets by country

### ✅ Step 3: Exploratory Data Analysis (EDA)
- Top 10 countries by confirmed cases, deaths, and death rate
- Top 10 happiest and least happy countries
- Correlation heatmaps for happiness indicators

### ✅ Step 4: Dataset Merging & Harmonization
- Merged happiness and COVID datasets on the country field
- Fixed mismatched country names where needed

### ✅ Step 5: Correlation Analysis & Visualizations
- Plotted scatter plots and heatmaps:
  - Happiness Score vs COVID-19 Death Rate
  - GDP per Capita vs Confirmed Cases
  - Life Expectancy vs Death Rate

### ✅ Step 6: Summary & Conclusions
- Derived insights from the merged dataset and plotted results

---

## 📊 Key Insights

- Countries with higher **Happiness Scores** generally had **lower COVID-19 death rates**
- **Life Expectancy** and **GDP per Capita** showed moderate negative correlation with COVID-19 death rates
- Countries with **strong social support and freedom** tended to show better pandemic resilience
- Some large countries like the USA were outliers due to population size and complex demographics

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository or download the `.ipynb` notebook
2. Ensure all datasets are in the same directory
3. Open the notebook in Jupyter and run cells sequentially

---

