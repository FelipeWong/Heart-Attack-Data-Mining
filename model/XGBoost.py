import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_evaluate_xgboost(data):
    # Set input and output
    x = data.drop('output', axis=1)  # Input features
    y = data['output']  # Target variable

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Define the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )

    # Train the XGBoost model
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate the model
    xg_accuracy = accuracy_score(y_test, y_pred)
    return xg_accuracy
