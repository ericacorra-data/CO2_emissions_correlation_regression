# Correlation and Regression Analysis of CO₂ Emissions in Vehicles
Analysis of factors influencing **CO₂ emissions** in vehicles (Canada) 
## Objective
- Identify correlations between features and CO₂ emissions.
- Build regression models to predict CO₂ emissions for unobserved vehicles.
- Explore and account for non-linear relationship using polynomial terms
## Dataset 
- Source: [Government of Canada – Fuel Consumption Ratings](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
- File used: `FuelConsumptionCo2.csv`
## Analysis and Methods
- Exploratory Data Analysis (EDA)
- Linear Regression (Simple Linear Regression, Multiple Linear Regression, Polynomial Regression (degree 2), Multiple Linear Regression with Polynomial Term)
- Residual Diagnostic
## Evaluation Metrics
- R² (train/test)
- Mean Squared Error (MSE)
- Residual analysis
- Actual vs Predicted plots
## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, scipy, sklearn
## Limitations
- Dataset limited to Canada; results may not generalize globally.
- Some categorical features were dropped for simplicity (e.g., vehicle make, model).
