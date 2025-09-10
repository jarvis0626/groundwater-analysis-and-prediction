from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

data_folder = "data"
all_data = pd.DataFrame()

# Load and preprocess data
for file in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file)
    if file.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file.endswith(".xlsx"):
        year = int(file.split('_')[1].split('.')[0])
        df_raw = pd.read_excel(file_path, header=None)
        column_names = df_raw.iloc[1].fillna("").astype(str)
        min_max_labels = df_raw.iloc[2].fillna("").astype(str)
        final_columns = ["State Name" if "State" in col else f"{col} {label}".strip() if label else col 
                         for col, label in zip(column_names, min_max_labels)]
        df_raw.columns = final_columns
        df = df_raw.iloc[3:].reset_index(drop=True)
        df['Year'] = year
    else:
        continue
    all_data = pd.concat([all_data, df], ignore_index=True) if not all_data.empty else df

# Clean column names
all_data.columns = all_data.columns.str.replace("\n", " ").str.strip()

expected_parameters = [
    "Temperature ⁰C", "pH", "Conductivity (µmhos/cm)", "B.O.D. (mg/l)",
    "Nitrate-N + Nitrite-N (mg/l)", "Faecal Coliform (MPN/100ml)", "Total Coliform (MPN/100ml)"
]

# Create average columns for min/max values
for col in all_data.columns:
    if "Min" in col:
        base_name = col.replace(" Min", "")
        max_col = base_name + " Max"
        if max_col in all_data.columns:
            all_data[base_name] = all_data[[col, max_col]].astype(float).mean(axis=1)

def classify_water_quality(row):
    ph_value = pd.to_numeric(row.get('pH Min'), errors='coerce')
    coliform = pd.to_numeric(row.get('Faecal Coliform (MPN/100ml) Min'), errors='coerce')
    nitrate = pd.to_numeric(row.get('Nitrate-N + Nitrite-N (mg/l) Min'), errors='coerce')
    if pd.isna(ph_value) or pd.isna(coliform) or pd.isna(nitrate):
        return "Unknown"
    return "Good" if 6.5 <= ph_value <= 8.5 and coliform < 10 and nitrate < 10 else "Poor"

all_data['Groundwater Quality'] = all_data.apply(classify_water_quality, axis=1)
encoder = LabelEncoder()
all_data['Groundwater Quality'] = encoder.fit_transform(all_data['Groundwater Quality'])

features = ['pH Min', 'Conductivity (µmhos/cm) Min', 'B.O.D. (mg/l) Min', 
            'Faecal Coliform (MPN/100ml) Min', 'Total Coliform (MPN/100ml) Min', 
            'Nitrate-N + Nitrite-N (mg/l) Min']

X = all_data[features].replace("BDL", np.nan).astype(float)

# Impute missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)
y = all_data['Groundwater Quality']



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply PCA separately to training and testing data
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Train model on PCA features
model = DecisionTreeRegressor(max_depth=4, min_samples_split=3, random_state=42)
model.fit(X_train_pca, y_train)

# Compute Feature Importance & VIF
feature_importance = dict(zip(features, model.feature_importances_))


@app.route('/')
def home():
    return render_template('index.html')

from sklearn.metrics import mean_squared_error, mean_absolute_error

@app.route('/get_model_metrics', methods=['GET'])
def get_model_metrics():
    train_preds = model.predict(X_train_pca)
    test_preds = model.predict(X_test_pca)
    
    return jsonify({
        "feature_importance": dict(zip(features, model.feature_importances_)),
        "train_r2_score": round(r2_score(y_train, train_preds), 4),
        "test_r2_score": round(r2_score(y_test, test_preds), 4),
        "train_mse": round(mean_squared_error(y_train, train_preds), 4),
        "test_mse": round(mean_squared_error(y_test, test_preds), 4),
        "train_mae": round(mean_absolute_error(y_train, train_preds), 4),
        "test_mae": round(mean_absolute_error(y_test, test_preds), 4)
    })


@app.route('/get_states', methods=['GET'])
def get_states():
    print("Available columns:", all_data.columns.tolist())  # Debugging step
    print("Sample data:", all_data.head().to_dict())  # Debugging step
    
    if 'State Name' not in all_data.columns:
        return jsonify({'error': 'State Name column not found'}), 500

    states = all_data['State Name'].dropna().unique().tolist()
    
    if not states:
        return jsonify({'error': 'No states found'}), 404

    return jsonify(states)


@app.route('/predict', methods=['POST'])
def predict():
    state = request.json.get('state', None)
    if not state:
        return jsonify({'error': 'No state provided'}), 400

    state_data = all_data[all_data['State Name'] == state].copy()
    if state_data.empty:
        return jsonify({'error': 'State not found'}), 404

    state_data[features] = state_data[features].apply(pd.to_numeric, errors='coerce')
    state_aggregated = state_data.groupby('Year', as_index=False)[features + ['Groundwater Quality']].mean()
    if state_aggregated.empty:
        return jsonify({'error': 'No valid data for the selected state'}), 400

    historical_years = state_aggregated['Year'].tolist()
    historical_quality = state_aggregated['Groundwater Quality'].tolist()
    future_years = [historical_years[-1] + i for i in range(1, 4)]

    future_quality = []
    latest_data = state_aggregated.iloc[-1][features].tolist()
    for _ in range(3):
        future_input = pd.DataFrame([latest_data], columns=features)
        future_input_imputed = imputer.transform(future_input)  # Use same imputer
        future_input_scaled = scaler.transform(future_input_imputed)  # Use same scaler
        future_input_pca = pca.transform(future_input_scaled)  # Apply PCA transformation
        future_quality.append(float(model.predict(future_input_pca)[0])) 

    # ✅ **Filter test data for the selected state**
    state_test_data = all_data[(all_data['State Name'] == state)]
    if state_test_data.empty:
        return jsonify({'error': 'No test data available for the selected state'}), 400

    X_state = state_test_data[features].replace("BDL", np.nan).astype(float)
    X_state_imputed = imputer.transform(X_state)
    X_state_scaled = scaler.transform(X_state_imputed)
    X_state_pca = pca.transform(X_state_scaled)

    y_state_actual = state_test_data['Groundwater Quality']
    y_state_pred = model.predict(X_state_pca)

    # ✅ **Compute Evaluation Metrics per state**
    metrics = {
        "rmse": round(mean_squared_error(y_state_actual, y_state_pred) ** 0.5, 4),
        "mae": round(mean_absolute_error(y_state_actual, y_state_pred), 4),
        "r2": round(r2_score(y_state_actual, y_state_pred), 4),
    }

    response_data = {
        'years': historical_years + future_years,
        'quality': historical_quality + future_quality,
        'avg_features': state_aggregated[features].mean().to_dict(),
        'feature_importance': feature_importance,
        'metrics': metrics  # ✅ **Now, metrics update for each state**
    }

    print("Test Predictions for", state, ":", y_state_pred)
    print("Evaluation Metrics for", state, ":", metrics)

    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True)