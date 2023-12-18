from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normalize_data(data, numerical_columns):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Perform normalization on the numerical columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data


def one_hot_encode(data, categorical_columns):
    # Perform one-hot encoding on the categorical columns
    data_encoded = pd.get_dummies(data, columns=categorical_columns)

    return data_encoded
