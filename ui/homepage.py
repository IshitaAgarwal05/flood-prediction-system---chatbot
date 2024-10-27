import flet as ft
import numpy as np
import pandas as pd
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import os
import webbrowser

# Load the dataset
data = pd.read_csv('Updated_Flood_Classification.csv')

# Create an imputer for handling missing values
imputer = SimpleImputer(strategy='mean')

# Define India's boundaries
INDIA_BOUNDS = {
    'min_lat': 8.4,
    'max_lat': 37.6,
    'min_lon': 68.7,
    'max_lon': 97.25
}

def validate_columns(data):
    # Check if required columns exist based on your actual column names
    required_columns = ['Dist_Name', 'Rainfall (mm/hr)', 'flood(Y)', 'Latitude', 'Longitude']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def is_within_india(lat, lon):
    return (INDIA_BOUNDS['min_lat'] <= lat <= INDIA_BOUNDS['max_lat'] and 
            INDIA_BOUNDS['min_lon'] <= lon <= INDIA_BOUNDS['max_lon'])

def find_nearest_location(lat, lon, data_df):
    if not is_within_india(lat, lon):
        return None
    
    min_distance = float('inf')
    nearest_idx = None
    
    for idx, row in data_df.iterrows():
        dist = geodesic((lat, lon), (row['Latitude'], row['Longitude'])).kilometers
        if dist < min_distance:
            min_distance = dist
            nearest_idx = idx
    
    return data_df.iloc[nearest_idx] if nearest_idx is not None else None

def create_heatmap(data_df):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    for _, row in data_df.iterrows():
        try:
            # Handle missing values in features
            features = np.array(row['Rainfall (mm/hr)']).reshape(1, -1)
            if np.isnan(features).any():
                features = imputer.transform(features)
            
            probability = best_model.predict_proba(features)[0][1]
            
            color = f'rgb({int(255*probability)}, {int(255*(1-probability))}, 0)'
            
            # Create popup content
            popup_content = f"Rainfall: {row['Rainfall (mm/hr)']} mm/hr\nProbability: {probability:.2f}"
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                popup=popup_content
            ).add_to(m)
            
        except Exception as e:
            print(f"Warning: Error processing row: {e}")
            continue
    
    map_path = "flood_map.html"
    m.save(map_path)
    return map_path

class MapView(ft.UserControl):
    def __init__(self, map_path):
        super().__init__()
        self.map_path = map_path
    
    def build(self):
        return ft.ElevatedButton(
            "View Map",
            on_click=lambda _: webbrowser.open('file://' + os.path.abspath(self.map_path))
        )
    
class MetricsView(ft.UserControl):
    def __init__(self, results):
        super().__init__()
        self.results = results

    def build(self):
        # Create headers
        headers = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Model")),
                ft.DataColumn(ft.Text("Accuracy")),
                ft.DataColumn(ft.Text("Precision")),
                ft.DataColumn(ft.Text("Recall")),
                ft.DataColumn(ft.Text("F1 Score"))
            ],
            rows=[]
        )

        # Add rows for each model
        for model_name, metrics in self.results.items():
            headers.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(model_name)),
                        ft.DataCell(ft.Text(f"{metrics['Accuracy']:.3f}")),
                        ft.DataCell(ft.Text(f"{metrics['Precision']:.3f}")),
                        ft.DataCell(ft.Text(f"{metrics['Recall']:.3f}")),
                        ft.DataCell(ft.Text(f"{metrics['F1 Score']:.3f}"))
                    ]
                )
            )

        # Find the best model
        best_model = max(self.results.items(), key=lambda x: x[1]['Accuracy'])
        best_model_text = ft.Text(
            f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.3f})",
            size=20,
            weight=ft.FontWeight.BOLD
        )

        return ft.Column([
            ft.Container(
                content=ft.Text("Model Performance Metrics", size=24, weight=ft.FontWeight.BOLD),
                margin=ft.margin.only(bottom=20)
            ),
            best_model_text,
            ft.Container(
                content=headers,
                margin=ft.margin.only(top=20),
                padding=10,
                border=ft.border.all(1, ft.colors.GREY_400),
                border_radius=10
            )
        ])

# Data preparation and model training
def prepare_data_and_train_models():
    global best_model, results, imputer
    
    # Drop rows where target variable is missing
    df = data.dropna(subset=['flood(Y)'])
    
    # Prepare features and target
    X = df[['Rainfall (mm/hr)']].values
    y = df['flood(Y)'].values
    
    # Fit the imputer on training data
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    results = {}
    best_accuracy = 0
    best_model = None
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            
        results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        }
    
    return results

def main(page: ft.Page):
    page.title = "Flood Prediction UI"
    page.theme_mode = ft.ThemeMode.LIGHT
    
    try:
        validate_columns(data)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Convert district names to lowercase in the dataset
    data['Dist_Name_Lower'] = data['Dist_Name'].str.lower()

    # Alert dialog for flood risk
    alert_dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text("⚠️ Flood Risk Alert!", size=20, color=ft.colors.RED),
        content=ft.Text(
            "This area is at high risk of flooding! Please take necessary precautions.",
            size=16
        ),
        actions=[
            ft.TextButton(text="OK", on_click=lambda e: close_alert(e))
        ]
    )

    def close_alert(e):
        alert_dialog.open = False
        page.update()

    # Input fields
    location_type = ft.RadioGroup(
        content=ft.Row([
            ft.Radio(value="name", label="Place Name"),
            ft.Radio(value="coords", label="Coordinates")
        ]),
        value="name"
    )
    
    place_input = ft.TextField(label="Enter Place Name")
    lat_input = ft.TextField(
        label="Latitude",
        disabled=True,
        value="0"
    )
    lon_input = ft.TextField(
        label="Longitude",
        disabled=True,
        value="0"
    )
    result_output = ft.Text(size=16)
    error_text = ft.Text(color=ft.colors.RED)

    def rail_change(e):
        index = e.control.selected_index
        main_content.content = views[index]
        page.update()

    # Initialize models and prepare data
    global results
    results = prepare_data_and_train_models()
    
    # Create the map
    map_path = create_heatmap(data)
    map_view = MapView(map_path)
    
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.icons.HOME_OUTLINED,
                selected_icon=ft.icons.HOME,
                label="Predict"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.ANALYTICS_OUTLINED,
                selected_icon=ft.icons.ANALYTICS,
                label="Metrics"
            ),
        ],
        on_change=rail_change
    )

    def show_alert():
        page.dialog = alert_dialog
        alert_dialog.open = True
        page.update()

    def location_type_changed(e):
        place_input.disabled = location_type.value == "coords"
        lat_input.disabled = location_type.value == "name"
        lon_input.disabled = location_type.value == "name"
        error_text.value = ""
        page.update()

    location_type.on_change = location_type_changed

    def on_predict(e):
        error_text.value = ""
        try:
            if location_type.value == "name":
                # Convert input to lowercase for case-insensitive comparison
                dist_name_lower = place_input.value.lower().strip()
                
                # Check if the lowercase district name exists in the dataset
                if dist_name_lower not in data['Dist_Name_Lower'].values:
                    error_text.value = f"District '{place_input.value}' not found in database"
                    page.update()
                    return
                
                # Get the original case district name and data
                location_data = data[data['Dist_Name_Lower'] == dist_name_lower].iloc[0]
                
            else:  # coordinates
                print(f"Debug - Lat input: '{lat_input.value}'")
                print(f"Debug - Lon input: '{lon_input.value}'")
                
                try:
                    lat = float(lat_input.value if lat_input.value else 0)
                    lon = float(lon_input.value if lon_input.value else 0)
                    
                    print(f"Debug - Parsed coordinates: lat={lat}, lon={lon}")
                    
                    if not is_within_india(lat, lon):
                        error_text.value = f"Coordinates must be within India's boundaries:\nLatitude: 8.4 to 37.6\nLongitude: 68.7 to 97.25"
                        page.update()
                        return
                    
                    location_data = find_nearest_location(lat, lon, data)
                    if location_data is None:
                        error_text.value = "No nearby locations found in database"
                        page.update()
                        return
                    
                    print(f"Debug - Found nearest location: {location_data['Dist_Name']}")
                    
                except ValueError as ve:
                    print(f"Debug - ValueError: {str(ve)}")
                    error_text.value = "Please enter valid numeric coordinates"
                    page.update()
                    return

            # Make prediction
            features = np.array(location_data['Rainfall (mm/hr)']).reshape(1, -1)
            if np.isnan(features).any():
                features = imputer.transform(features)
                
            prediction = best_model.predict(features)
            probability = best_model.predict_proba(features)[0][1]

            result_output.value = (
                f"Prediction for {location_data['Dist_Name']}:\n"
                f"Location: (Lat: {location_data['Latitude']:.4f}, Lon: {location_data['Longitude']:.4f})\n"
                f"Rainfall: {location_data['Rainfall (mm/hr)']:.2f} mm/hr\n"
                f"Flood Risk: {'Yes' if prediction[0] == 1 else 'No'}\n"
                f"Probability: {probability:.2f}"
            )
            
            # Show alert if flood risk is high
            if prediction[0] == 1:
                show_alert()
            
        except Exception as e:
            print(f"Debug - Exception details: {str(e)}")
            error_text.value = f"An error occurred: {str(e)}"
        
        page.update()

    predict_button = ft.ElevatedButton("Predict", on_click=on_predict)
    
    # Create views
    predict_view = ft.Column([
        location_type,
        place_input,
        ft.Row([lat_input, lon_input]),
        predict_button,
        error_text,
        result_output,
        map_view
    ])

    metrics_view = MetricsView(results)
    views = [predict_view, metrics_view]
    main_content = ft.Container(content=views[0], expand=True)
    
    page.add(
        ft.Row([
            rail,
            ft.VerticalDivider(width=1),
            main_content
        ], expand=True)
    )    

# Prepare the data and train models
df = data.dropna(subset=['Rainfall (mm/hr)', 'flood(Y)'])

# Define independent (X) and dependent (Y) variables
X = df[['Rainfall (mm/hr)']].dropna().values
Y = df['flood(Y)'].dropna().values

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
best_accuracy = 0
best_model = None

for model_name, model in models.items():
    if model_name=="Random Forest":
        results_rf = []

        for n_trees in range(1, 101):
            rf_model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            rf_model.fit(X_train, y_train)        
            y_pred = rf_model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results_rf.append((n_trees, accuracy))

        results_df = pd.DataFrame(results_rf, columns=['n_trees', 'accuracy'])
        best_n_trees = results_df.loc[results_df['accuracy'].idxmax()]['n_trees']
        
        # Fit the final model with the best number of trees
        final_rf_model = RandomForestClassifier(n_estimators=int(best_n_trees), random_state=42)
        final_rf_model.fit(X_train, y_train)
        y_pred_final = final_rf_model.predict(X_test)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0)
    }

ft.app(target=main)