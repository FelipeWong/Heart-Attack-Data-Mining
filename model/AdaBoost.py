from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_evaluate_adaboost(data):
    # Step 2: Prepare your data
    x = data.drop('output', axis=1)  # Input features
    y = data['output']  # Target variable

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # Step 3: Define the AdaBoost model
    model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0
    )

    # Step 4: Train the AdaBoost model
    model.fit(x_train, y_train)

    # Step 5: Make predictions
    y_pred = model.predict(x_test)

    # Step 6: Evaluate the model
    ada_accuracy = accuracy_score(y_test, y_pred)
    return ada_accuracy
