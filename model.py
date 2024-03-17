import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\deepa\Downloads\trian.csv")

data['purchase_date'] = pd.to_datetime(data['purchase_date'])

data['year'] = data['purchase_date'].dt.year
data['month'] = data['purchase_date'].dt.month
data['day'] = data['purchase_date'].dt.day

data.drop(['purchase_date', 'product_id'], axis=1, inplace=True)

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode categorical variables
data['region'] = label_encoder.fit_transform(data['region'])

# Split data into features (X) and target variable (y)
X = data.drop('region', axis=1)
y = data['region']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model pipeline
numeric_features = ['day', 'month', 'year']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Append regressor to preprocessing pipeline.
# Now we have a full prediction pipeline.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMS value
rms = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rms)

# Function to predict quantity for given date and region
def predict_quantity(date_str, region):
    date = datetime.strptime(date_str, '%m-%d-%Y')
    year = date.year
    month = date.month
    day = date.day
    input_data = [[year, month, day]]  # Modified to match the model input
    input_df = pd.DataFrame(input_data, columns=['year', 'month', 'day'])  # Convert list to DataFrame
    predicted_quantity = model.predict(input_df)
    return predicted_quantity

# User input for date and region
date_input = input("Enter the date (MM-DD-YYYY format): ")
region_input = input("Enter the region: ")

# Predict quantity based on user input
predicted_quantity = predict_quantity(date_input, region_input)
print('Predicted quantity:', predicted_quantity)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Region')
plt.legend()
plt.grid(True)
plt.show()