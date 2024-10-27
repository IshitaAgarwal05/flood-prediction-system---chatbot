import flet as ft
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Sample dataset loading and preprocessing (replace this with your actual dataset)
data = pd.DataFrame({
    'Place': ['Place1', 'Place2', 'Place3', 'Place4'],
    'Feature1': [1, 2, 3, 4],
    'Feature2': [10, 20, 30, 40],
    'Flooded': [0, 1, 1, 0]
})

# Encoding categorical variable
data['Flooded'] = data['Flooded'].astype(int)

# Split dataset into features and target
X = data[['Feature1', 'Feature2']]  # Replace with actual feature columns
y = data['Flooded']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)  # Updated to handle undefined precision
    recall = recall_score(y_test, y_pred, zero_division=0)        # Updated to handle undefined recall
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# Flet UI code
def main(page: ft.Page):
    page.title = "Flood Prediction UI"

    # Input for place name
    place_input = ft.TextField(label="Enter Place Name")
    result_output = ft.Text()

    def on_predict(e):
        # Here you would implement logic to retrieve features based on the place name.
        feature_values = np.array([[2, 20]])  # Replace with actual feature extraction logic
        
        # Make predictions using the best model
        best_model = RandomForestClassifier(n_estimators=100)  # Or the model with best results
        best_model.fit(X_train, y_train)  # Fit the model again if necessary

        prediction = best_model.predict(feature_values)
        probability = best_model.predict_proba(feature_values)[0][1]  # Probability of being flooded

        flood_status = "Yes" if prediction[0] == 1 else "No"
        result_output.value = f"Flooded: {flood_status} (Probability: {probability:.2f})"
        page.update()

    predict_button = ft.ElevatedButton("Predict", on_click=on_predict)

    # Create a results table
    results_table = ft.DataTable(columns=[
        ft.DataColumn(ft.Text("Model")),
        ft.DataColumn(ft.Text("Accuracy")),
        ft.DataColumn(ft.Text("Precision")),
        ft.DataColumn(ft.Text("Recall")),
    ])
    
    for model_name, metrics in results.items():
        results_table.rows.append(ft.DataRow(cells=[
            ft.DataCell(ft.Text(model_name)),
            ft.DataCell(ft.Text(f"{metrics['Accuracy']:.2f}")),
            ft.DataCell(ft.Text(f"{metrics['Precision']:.2f}")),
            ft.DataCell(ft.Text(f"{metrics['Recall']:.2f}")),
        ]))

    page.add(place_input, predict_button, result_output, results_table)

ft.app(target=main)
