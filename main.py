import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
# for polynomial regression :
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# loading data
data = pd.read_csv('ElecDeviceRatingPrediction.csv')
# Replace 'Not Available' values with the mode
mode_value = data['processor_gnrtn'].mode()[0]
data['processor_gnrtn'] = data['processor_gnrtn'].replace('Not Available', mode_value)

null_counts = data.isnull().sum()
# print(null_counts)

# nulls values
data.dropna(axis=0, how='any', inplace=True)
# print(data)

# encoding

# Perform one hot encoding
oneHotEncoded = pd.get_dummies(data['processor_brand'], prefix='processor_brand')
encodedInt = oneHotEncoded.astype(int)
dataEncoded = pd.concat([data, encodedInt], axis=1)
dataEncoded.drop('processor_brand', axis=1, inplace=True)

# perform LabdelEncoder
encodedColumns = ['brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type',
                  'ssd', 'hdd', 'os', 'graphic_card_gb', 'weight', 'warranty',
                  'Touchscreen', 'msoffice', 'rating']
label_encoder = LabelEncoder()

for column in encodedColumns:
    dataEncoded[column] = label_encoder.fit_transform(dataEncoded[column])

# outliers handling
sns.boxplot(dataEncoded, palette="rainbow", orient='h')


# plt.show()

def remove_outliers(XEncoded):
    for column in XEncoded.columns:
        if XEncoded[column].nunique() > 2:  # for binary data / categorical
            Q1 = XEncoded[column].quantile(0.25)
            Q3 = XEncoded[column].quantile(0.75)
            IQR = Q3 - Q1
            lowerlimit = Q1 - (1.5 * IQR)
            upperlimit = Q3 + (1.5 * IQR)
            XEncoded.loc[(XEncoded[column] < lowerlimit) | (XEncoded[column] > upperlimit), column] = np.nan
    XEncoded.fillna(XEncoded.median().iloc[0], inplace=True)
    return XEncoded


dataEncoded = remove_outliers(dataEncoded)

sns.boxplot(dataEncoded, palette="rainbow", orient='h')
# plt.show()

scaler = MinMaxScaler()
Scaled = scaler.fit_transform(dataEncoded)
Scaled = pd.DataFrame(Scaled, columns=dataEncoded.columns)

# correlation
correlation = dataEncoded.corr()
plt.figure(figsize=(20, 15))
plot = sns.heatmap(correlation.round(3), annot=True)
plot.set_title("correlation")
# plt.show()

# feature selection(Normalization)
topFeature = correlation.index[abs(correlation['rating']) > 0.1]
topFeature = topFeature.drop('rating', errors='ignore')
Selected = Scaled[topFeature]

X = Scaled.drop(columns=['rating'])
Y = Scaled['rating']
# ## Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=10)

# Initialize RandomForestRegressor as the estimator
estimator = RandomForestRegressor()
# Initialize RFE
rfe = RFE(estimator, n_features_to_select=5)
# Fit RFE
rfe.fit(X_train, Y_train)
# Get the selected features
selected_feature = X.columns[rfe.support_]
# support_print("Selected Features:")
# print(selected_feature)


# POLYNOMIAL REGRESSION
# START
print(" POLYNOMIAL REGRESSION MODEL\n")
selected_features = selected_feature

# print(selected_features)
while True:
    # user input
    degree = int(input(" Enter the degree of the polynomial regression : "))

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train[selected_features])

    # fitting step
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, Y_train)

    Y_train_pred = poly_reg.predict(X_poly)

    X_test_poly = poly.transform(X_test[selected_features])
    Y_test_pred = poly_reg.predict(X_test_poly)

    # mse
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    print(" Mean square error on training set : ", train_mse)
    print(" Mean square error on test set : ", test_mse)

    # loop break
    choice = input(" Do you want to continue ? (y/n): ").upper()
    if choice != 'Y':
        break
# END

print(
    "_")

# LINEAR REGRESSION
# START
print(" LINEAR REGRESSION MODEL\n")
# 1. Create the Linear Regression Model
model = LinearRegression()

# 2. Train the Model
model.fit(X_train, Y_train)

# 3. Make Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 4. Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(Y_train, y_pred_train)
mse_test = mean_squared_error(Y_test, y_pred_test)

# 5. Print the Results
print(f" Mean Squared Error (MSE) for training data: {mse_train}")
print(f" Mean Squared Error (MSE) for test data: {mse_test}")

# END
