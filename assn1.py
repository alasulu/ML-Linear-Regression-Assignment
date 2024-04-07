""" Code is written by Oğuzhan Alasulu, Emirhan Şahinoğlu and Recep Akpunar"""


#--- Data Import ---#
import pandas as pd
DataFrame = pd.read_csv('BioSD student dataset 2023.csv') # Importing First Dataset
DataFrame2 = pd.read_csv('BioSD student dataset 2024.csv') # Importing Second Dataset


#--- Data Cleaning ---#
# Missing values, Duplicates, and Inconsistencies in First Data Frame 
requested_value = 155
DataFrame.loc[4, 'Please enter your height (cm)'] = requested_value
requested_value2 = 176
DataFrame.loc[15, 'Please enter your height (cm)'] = requested_value2

DataFrame['Please enter your hometown (city).'] = DataFrame['Please enter your hometown (city).'].replace({
    'kayseri': 'Kayseri',
    'Elaziğ': 'Elazığ',
    'Istanbul': 'İstanbul',
    'izmir': 'İzmir',
    'Izmir': 'İzmir',
    'Kauseri': 'Kayseri',
    'Kirikkale ': 'Kırıkkale', 
    'ISTANBUL': 'İstanbul'
})

print(DataFrame) # Printing the DataFrame


# Missing values, Duplicates, and Inconsistencies in Second Data Frame 
requested_value3 = 177
DataFrame2.loc[4, 'Please enter your height (cm)'] = requested_value3
DataFrame2['Please enter your height (cm)'] = DataFrame2['Please enter your height (cm)'].replace({
    1.6: 160  # Corrected the column name to 'Please enter your height (cm)'
})
DataFrame2['Please enter your home town (city).'] = DataFrame2['Please enter your home town (city).'].replace({
    'konya': 'Konya'
})
DataFrame2['Please enter your birth year:'] = DataFrame2['Please enter your birth year:'].replace({
    '05.11.1999': '1999',
    '07.03.2001': '2001',
    '23.03.2001': '2001'
})
requested_value4 = 2001
DataFrame2.loc[8, 'Please enter your birth year:'] = requested_value4

print(DataFrame2) # Printing the DataFrame


#--- Data Exploration ---#
# First Data Frame
print("First few rows of the dataset:")
print(DataFrame.head()) # Display the first few rows of the dataset
print("\nSummary statistics of the dataset:")
print(DataFrame.describe()) # Summary statistics of the dataset
print("\nInformation about the dataset:")
print(DataFrame.info()) # Information about the dataset
print("\nMissing values in the dataset:")
print(DataFrame.isnull().sum()) # Check for missing values
print("\nNumber of duplicate rows in the dataset:", DataFrame.duplicated().sum())
# Check for duplicates

# Second Data Frame
print("First few rows of the dataset:")
print(DataFrame2.head()) # Display the first few rows of the dataset
print("\nSummary statistics of the dataset:")
print(DataFrame2.describe()) # Summary statistics of the dataset
print("\nInformation about the dataset:")
print(DataFrame2.info()) # Information about the dataset
print("\nMissing values in the dataset:") 
print(DataFrame2.isnull().sum()) # Check for missing value
print("\nNumber of duplicate rows in the dataset:", DataFrame2.duplicated().sum())
# Check for duplicates


#--- Data Visualization ---#
import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for height distribution in both datasets
plt.figure(figsize=(10, 6))
sns.histplot(DataFrame['Please enter your height (cm)'], color='blue', alpha=0.5, label='2023')
sns.histplot(DataFrame2['Please enter your height (cm)'], color='red', alpha=0.5, label='2024')
plt.title('Height Distribution in 2023 and 2024')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# Box plot for height distribution in both datasets
plt.figure(figsize=(10, 6))
sns.boxplot(x='Year', y='Please enter your height (cm)', data=pd.concat([DataFrame.assign(Year='2023'), DataFrame2.assign(Year='2024')]))
plt.title('Height Distribution in 2023 and 2024')
plt.xlabel('Year')
plt.ylabel('Height (cm)')
plt.show()


#--- Linear Regression ---#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Linear Regression for Data Frame 1
# Convert birth year to numeric
DataFrame['Please enter your weight (kg)'] = pd.to_datetime(DataFrame['Please enter your height (cm)']).dt.year.astype(float)
# Extracting features and target variable
X = DataFrame[['Please enter your weight (kg)']]  # Feature
y = DataFrame['Please enter your height (cm)']    # Target variable
poly_features = PolynomialFeatures(degree=2)  # Polinomik özellikler
X_poly = poly_features.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42) # Splitting the dataset into training and testing sets
model = LinearRegression()
model.fit(X_train, y_train) # Training the linear regression model
y_pred = model.predict(X_test) # Predicting on the testing set
mse = mean_squared_error(y_test, y_pred) # Evaluating the model's performance
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Linear Regression for Data Frame 2
# Convert birth year to numeric
DataFrame2['Please enter your birth year:'] = pd.to_datetime(DataFrame2['Please enter your birth year:']).dt.year.astype(float)
# Extracting features and target variable
X = DataFrame2[['Please enter your birth year:']]  # Feature
y = DataFrame2['Please enter your height (cm)']    # Target variable
poly_features = PolynomialFeatures(degree=2)  # Polinomik özellikler
X_poly = poly_features.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42) # Splitting the dataset into training and testing sets
model = LinearRegression()
model.fit(X_train, y_train) # Training the linear regression model
y_pred = model.predict(X_test) # Predicting on the testing set
mse = mean_squared_error(y_test, y_pred) # Evaluating the model's performance
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)