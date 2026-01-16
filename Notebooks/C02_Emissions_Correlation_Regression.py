#!/usr/bin/env python
# coding: utf-8

# # Correlation and Regression Analysis of CO₂ Emissions in Vehicles

# ## 1. OBJECTIVES: 
# Identifying correlation between features and finding the best regression model to predict CO2 emissions (target variable) of unobserved cars based on the selected features.

# In[1]:


# Import Python libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# ## 2. DATASET AND METHOD
# The used data is a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
# * You can download it here: "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv" (dataset included in the IBM Data Science Professional Certificate)
# 

# In[2]:


# read  into a pandas dataframe:
df_fuel=pd.read_csv("FuelConsumptionCo2.csv")
display(df_fuel.head())


# Each row consists of 13 features:
# - **MODEL YEAR** e.g. 2014
# - **MAKE** e.g. VOLVO
# - **MODEL** e.g. S60 AWD
# - **VEHICLE CLASS** e.g. COMPACT
# - **ENGINE SIZE** e.g. 3.0
# - **CYLINDERS** e.g 6
# - **TRANSMISSION** e.g. AS6
# - **FUEL TYPE** e.g. **Z**(Gasoline), **E**(alternative-fuel vehicles), **X**(E85:85% ethanol + gasoline mix), **D**(Diesel)
# - **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 13.2
# - **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 9.5
# - **FUEL CONSUMPTION COMBINED (L/100 km)** e.g. 11.5
# - **FUEL CONSUMPTION COMBINED MPG (MPG)** e.g. 25
# - **CO2 EMISSIONS (g/km)** e.g. 182 
# 

# In[3]:


df_fuel.shape


# In[4]:


## Check columns
print(df_fuel.columns)


# In[5]:


# check missing values
print(df_fuel.isnull().sum()) 


# In[6]:


print(df_fuel.info())


# In[7]:


# 'MAKE', 'MODEL', 'VEHICLECLASS','TRANSMISSION', 'FUELTYPE', dtype='object' : categorical variables


# In[8]:


# for numerical variables
df_fuel.describe().round(2)


# ## CO2 EMISSIONS by VEHICLECLASS

# In[9]:


plt.figure(figsize=(12, 6)) 
sns.boxplot(x="VEHICLECLASS", y="CO2EMISSIONS", data=df_fuel,  hue="VEHICLECLASS", palette="Set2", dodge=False)
plt.title("CO2 emissions by vehicleclass", fontsize=14)
plt.xlabel("Vehicleclass", fontsize=12)
plt.ylabel("CO2 Emissions (g/km)", fontsize=12)

plt.xticks(rotation=45, ha='right')  
plt.grid(axis='y', linestyle='--', alpha=0.7)  

plt.tight_layout()  
plt.show()


# * Van-Passenger has the highest CO2EMISSIONS compared with the other VEHICLECLASS.

# ## 3. CORRELATION

# In[10]:


# Drop categoricals and any useless columns like MODELYEAR which is the same for all cars
df_fuel = df_fuel.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
df_fuel.sample(5)


# In[11]:


df_fuel.corr()


# In[12]:


plt.figure(figsize=(8,6))
sns.heatmap(df_fuel.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# * The correlation matrix  shows that each variable presents a fairly high level of correlation with the target CO2EMISSIONS.

# In[13]:


corr_results = []

for col in df_fuel.columns:
    if col != "CO2EMISSIONS":
        r, p = pearsonr(df_fuel[col], df_fuel["CO2EMISSIONS"])
        corr_results.append([col, r, p])

corr_df = pd.DataFrame(
    corr_results,
    columns=["Feature", "Pearson r", "p-value"]
).sort_values(by="Pearson r", key=abs, ascending=False)

pd.options.display.float_format = '{:.6f}'.format
corr_df


# Although strong correlations are observed, correlation alone does not imply causation. 

# ## Linear Regression
# ### CO2 EMISSIONS vs ENGINESIZE

# In[14]:


plt.figure(figsize=(6,4))
sns.scatterplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df_fuel)
sns.regplot(x='ENGINESIZE', y='CO2EMISSIONS', data=df_fuel, scatter=False, color='red')
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("CO2 Emissions vs Engine Size")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[15]:


slope,intercept,r_value,p_value,std_err=linregress(df_fuel['ENGINESIZE'], df_fuel['CO2EMISSIONS'])
print("=== Hypothesis test ===")
print(f"Slope coefficient (b):{slope:.4f} g/km")
print(f"Intercept (a):{intercept:.4f}")
print(f"Correlation coefficient (r): {r_value:.4f}")
print(f"p-value:{p_value:.4f}")


# In[16]:


##  Inferential decision
alpha=0.05
if p_value < alpha:
    print("\n Result: Reject H0 -> the relationship is statistically significant.")
else: 
    print("\n Result: We fail to reject the null hypothesis (H0) -> no statistically significant relationship is observed.")


# * The hypothesis test confirms a strong and statistically significant correlation between ENGINESIZE and CO2EMISSIONS. The linear regression shows a positive and statistically significant relationship between the two variables (b = 39.12 g/km, r = 0.87, p < 0.001), indicating that the observed association is not attributable to chance.

# ### Predict CO2 Emissions from Engine Size

# In[17]:


# Select only the most relevant feature
X = df_fuel[['ENGINESIZE']]  # 2D array for sklearn
y = df_fuel['CO2EMISSIONS']  # target


# In[18]:


# Split into train and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[19]:


# Create linear regression model
lin_reg_simple = linear_model.LinearRegression()
lin_reg_simple .fit(X_train, y_train)


# In[20]:


# Coefficients
print("Coefficient:", lin_reg_simple .coef_[0])
print("Intercept:", lin_reg_simple .intercept_)


# In[21]:


# Predict on test set
y_pred = lin_reg_simple .predict(X_test)


# In[22]:


# Performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² score: {r2:.3f}")


# * The model achieves an R² of 0.762, meaning that ENGINESIZE alone accounts for about 76% of the variability in  CO2EMISSIONS. The MSE of 985.94 indicates non-negligible prediction errors. These results suggest that while ENGINESIZE is a strong predictor, additional variables are likely needed to improve predictive accuracy.

# In[23]:


# Plot regression line
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='lightblue', alpha=0.8, label='Data points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title("CO2 Emissions vs Engine Size")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[24]:


residuals = y_test - y_pred

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(alpha=0.3)
plt.show()


# * The residuals are generally distributed around zero, meaning that the model captures the linear relationship reasonably well. However, some residuals show relatively large deviations locally, suggesting the presence of non-negligible prediction errors. This is consistent with the previous results.

# In[25]:


plt.figure(figsize=(5,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Actual CO2")
plt.ylabel("Predicted CO2")
plt.title("Actual vs Predicted CO2 Emissions")
plt.grid(alpha=0.3)
plt.show()


# * The Actual vs Predicted values are largely concentrated around the line, indicating a good overall fit. However, for higher CO2EMISSIONS levels, some points deviates from the line, suggesting that the model tends to be less accurate for vehicles with high emissions. 

# ## Multiple linear regression

# Although VIF (Variance Inflation Factor) was not explicitly calculated, correlation analysis suggests that, to reduce multicollinearity, highly correlated predictors should be removed based on Pearson correlation.
# * 'ENGINESIZE' and 'CYLINDERS' are highly correlated, but 'ENGINESIZE' is more correlated with the target, so 'CYLINDERS' can be dropped.
# 
# * Each of the four fuel economy variables is highly correlated with each other. Since FUELCONSUMPTION_COMB_MPG is the most correlated with CO2EMISSIONS, the others can be dropped: 'FUELCONSUMPTION_CITY,' 'FUELCONSUMPTION_HWY,' 'FUELCONSUMPTION_COMB.'

# In[26]:


df_fuel = df_fuel.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
df_fuel.head(5)


# In[27]:


# Creation of a scatter matrix, which shows the scatter plots for each pair of input features. The diagonal of the matrix shows 
# each feature's histogram.
axes = pd.plotting.scatter_matrix(df_fuel, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()


# The relationship between FUELCONSUMPTION_COMB_MPG and CO2EMISSIONS is non-linear.

# In[28]:


# Extract the required columns and convert the resulting dataframes to NumPy arrays.
X = df_fuel.iloc[:,[0,1]].to_numpy()
y = df_fuel.iloc[:,[2]].to_numpy()


# In[29]:


# standardize the input features to ensure comparability of coefficients and numerical stability 
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)


# In[30]:


pd.DataFrame(X_std).describe().round(2)


# In[31]:


# Randomly split the data into train and test sets, using 80% of the dataset for training and reserving the remaining 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)


# In[32]:


# create a model object
lin_reg_multiple  = linear_model.LinearRegression()

# train the model in the training data
lin_reg_multiple.fit(X_train, y_train)

# Print the coefficients
coef_ =  lin_reg_multiple.coef_
intercept_ = lin_reg_multiple.intercept_

print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)


# In[33]:


# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)


# You would expect that for the limiting case of zero ENGINESIZE and zero FUELCONSUMPTION_COMB_MPG, the resulting CO2EMISSIONS should also be zero. This is inconsistent with the intercept of 329 g/km. The non-zero intercept does not have a physical interpretation, as zero engine size and zero fuel consumption are outside the observed data range. This does not invalidate the model, but highlights the limitations of linear extrapolation.
# 

# In[34]:


y_train_pred = lin_reg_multiple.predict(X_train)
y_test_pred = lin_reg_multiple.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R² train: {r2_train:.3f}")
print(f"R² test : {r2_test:.3f}")


# * The model explains approximately 88% of the variance in CO2EMISSIONS on both training and test data. The consistency between train and test R² values indicates a stable model with good generalization capability and limited overfitting. Compared to the simple linear regression using only ENGINESIZE, the multiple linear regression model significantly improves predictive performance. The test R² increases from 0.762 to 0.887, indicating that including additional relevant predictors captures more of the variability in CO2EMISSIONS.

# In[35]:


plt.figure(figsize=(5,5))

sns.scatterplot(
    x=y_test.flatten(),
    y=y_test_pred.flatten(),
    alpha=0.4,
    color="green"
)

# Linea og perfect prediction
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val],
         [min_val, max_val],
         'r--',
         linewidth=2,
         label='Perfect prediction')

plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Multiple Linear Regression: Actual vs Predicted CO2 Emissions")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# The Actual vs Predicted scatter plot shows a higher density of observations close to the 45-degree reference line, indicating good agreement between predicted and observed CO2EMISSIONS. Minor deviations are observed at higher and lower emission levels, but overall dispersion is reduced compared to the linear regression.

# In[36]:


# Predictions from the linear multiple regression model
y_test_pred_linear = lin_reg_multiple.predict(X_test).ravel()
y_test_true = y_test.ravel()

# Residuals
residuals_linear = y_test_true - y_test_pred_linear


plt.figure(figsize=(7,5))
sns.scatterplot(
    x=y_test_pred_linear,
    y=residuals_linear,
    alpha=0.5
)

plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values (Predicted CO2)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (Linear Multiple Regression)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# The residuals display a visible curvature pattern, indicating that the linearity assumption is violated and motivating the introduction of a polynomial term.

# ### CO2EMISSIONS vs FUELCONSUMPTION_COMB_MPG

# In[37]:


plt.figure(figsize=(8,5))
sns.scatterplot(x='FUELCONSUMPTION_COMB_MPG', y='CO2EMISSIONS', data=df_fuel)
sns.regplot(x='FUELCONSUMPTION_COMB_MPG', y='CO2EMISSIONS', data=df_fuel, scatter=False, color='red')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("CO2 Emissions vs Fuel MPG")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[38]:


slope,intercept,r_value,p_value,std_err=linregress(df_fuel['FUELCONSUMPTION_COMB_MPG'], df_fuel['CO2EMISSIONS'])
print("=== Hypothesis test ===")
print(f"Slope coefficient (b):{slope:.4f} g/km")
print(f"Intercept (a):{intercept:.4f}")
print(f"Correlation coefficient (r): {r_value:.4f}")
print(f"p-value:{p_value:.4f}")


# In[39]:


##  Inferential decision
alpha=0.05
if p_value < alpha:
    print("\n Result: Reject H0 -> the relationship is statistically significant.")
else: 
    print("\n Result: We fail to reject the null hypothesis (H0) -> no statistically significant relationship is observed.")


# * The hypothesis test reveals a strong and statistically significant negative association between combined FUELCONSUMPTION_COMB_MPG and CO2EMISSIONS. The estimated slope indicates that an increase of one MPG is associated, on average, with a reduction of approximately 7.7 g/km of CO2EMISSIONS. The strong negative association suggests a statistically significant relationship

# ### Polynomial regression 

# In[40]:


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df_fuel[["FUELCONSUMPTION_COMB_MPG"]])

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

lin_reg_poly = linear_model.LinearRegression()
lin_reg_poly.fit(X_train, y_train)

y_pred_poly = lin_reg_poly.predict(X_test)

print("R² (Polynomial):", r2_score(y_test, y_pred_poly))


# In[41]:


# Scatterplot of actual vs predicted CO2 emissions
plt.figure(figsize=(6,5))

# Scatter of actual points
plt.scatter(df_fuel['FUELCONSUMPTION_COMB_MPG'], df_fuel['CO2EMISSIONS'], color='lightblue', alpha=0.5, label='Actual Values')

# Sort data for a smooth line
X_sorted = np.sort(df_fuel['FUELCONSUMPTION_COMB_MPG']).reshape(-1,1)
y_poly_sorted = lin_reg_poly.predict(poly.transform(X_sorted))

# Plot predicted polynomial line
plt.plot(X_sorted, y_poly_sorted, color='red', linewidth=2, label='Polynomial Fit (Degree 2)')

plt.title("Polynomial Regression: CO2 Emissions vs Fuel MPG")
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Polynomial regression (degree 2) captures non-linear effects between FUELCONSUMPTION_COMB_MPG and CO2EMISSIONS. 

# ## Multiple Linear Regression with Polynomial Term (Corrected Model)

# In[42]:


# Create polynomial term for MPG
df_fuel["MPG_sq"] = df_fuel["FUELCONSUMPTION_COMB_MPG"] ** 2

df_fuel[["ENGINESIZE", "FUELCONSUMPTION_COMB_MPG", "MPG_sq", "CO2EMISSIONS"]].head()


# In[43]:


# definition of X and y 
X = df_fuel[["ENGINESIZE", "FUELCONSUMPTION_COMB_MPG", "MPG_sq"]]
y = df_fuel["CO2EMISSIONS"]


# In[44]:


# features standardization 
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[45]:


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)


# In[46]:


# model training
lin_reg_poly_mult = linear_model.LinearRegression()
lin_reg_poly_mult.fit(X_train, y_train)


# In[47]:


coef_std = lin_reg_poly_mult.coef_
intercept_std = lin_reg_poly_mult.intercept_

for name, coef in zip(X.columns, coef_std):
    print(f"{name}: {coef:.3f}")

print("Intercept:", intercept_std)


# - ENGINESIZE has a positive and statistically meaningful contribution to CO2EMISSIONS, even after controlling for fuel efficiency.
# - FUELCONSUMPTION_COMB_MPG shows a strong negative effect.
# - The positive quadratic term indicates diminishing marginal reductions in CO2EMISSIONS at higher FUELCONSUMPTION_COMB_MPG values, confirming the presence of a non-linear relationship between fuel efficiency and emissions.
# - The intercept has no physical interpretation, as zero engine size and zero fuel efficiency are outside the observed data domain.

# In[48]:


# model evaluation
y_train_pred = lin_reg_poly_mult.predict(X_train)
y_test_pred = lin_reg_poly_mult.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"R² train: {r2_train:.3f}")
print(f"R² test : {r2_test:.3f}")


# The polynomial multiple regression model achieves strong and consistent performance, with an R² of 0.88 on the training set and ~0.90 on the test set. The close agreement between train and test scores indicates good generalization and limited overfitting. Combined with improved residual behavior, these results suggest that the model captures the underlying data structure more accurately than the purely linear specification.

# In[49]:


# Actual vs Predicted
plt.figure(figsize=(5,5))

sns.scatterplot(
    x=y_test,
    y=y_test_pred,
    alpha=0.4,
    color="purple"
)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())

plt.plot([min_val, max_val],
         [min_val, max_val],
         'r--',
         linewidth=2,
         label='Perfect prediction')

plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Polynomial Multiple Regression: Actual vs Predicted CO2")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# The model shows improved predictive accuracy for vehicles with lower CO₂ emissions, as evidenced by the higher density of points close to the perfect prediction line.

# In[50]:


residuals = y_test - y_test_pred

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Polynomial Multiple Regression)")
plt.grid(alpha=0.3)
plt.show()


# ## CONCLUSIONS:
# - ENGINESIZE is strongly positively correlated with CO2EMISSIONS.
# - FUELCONSUMPTION_COMB_MPG presents a strong negative non-linear relationship with CO2EMISSIONS; higher FUELCONSUMPTION_COMB_MPG is associated with lower CO2EMISSIONS.
# - Simple linear regression (ENGINESIZE) explains ~76% of variability; multiple regression (ENGINESIZE + FUELCONSUMPTION_COMB_MPG) improves R² to ~88%, showing better predictive power.
# - Polynomial regression captures non-linear effects between FUELCONSUMPTION_COMB_MPG and CO2EMISSIONS, improving fit for extreme values.
# - By introducing a polynomial term for FUELCONSUMPTION_COMB_MPG (MPG²), the multiple linear regression model correctly captures the non-linear relationship between FUELCONSUMPTION_COMB_MPG and CO2EMISSIONS while maintaining interpretability. This specification improves R² to ~90%, and enhances predictive performance compared to the purely linear model.
# 

# In[ ]:




