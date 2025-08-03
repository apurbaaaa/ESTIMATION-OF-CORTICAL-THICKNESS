# task.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Target regions for prediction
target_names = [
    "ST58TA", "ST117TA", "ST40TA", "ST99TA", "ST32TA", "ST91TA",
    "ST60TA", "ST119TA", "ST62TA", "ST121TA"
]

def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_data.csv')
    df = pd.read_csv(csv_path)

    df['PTDOB'] = pd.to_datetime(df['PTDOB'])
    df['year'] = df['PTDOB'].dt.year
    df['month'] = df['PTDOB'].dt.month
    df['day'] = df['PTDOB'].dt.day
    df = df.drop(columns=['PTDOB'])

    X = df.drop(columns=target_names + ['RID'])
    y = df[target_names]

    # Outlier removal
    cols = ['APVOLUME', 'ABETA42', 'TAU', 'PTAU', 'PLASMAPTAU181']
    Q1 = X[cols].quantile(0.25)
    Q3 = X[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X[cols] < (Q1 - 1.5 * IQR)) | (X[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)

    X_parts = np.array_split(X_scaled, 4)
    y_parts = np.array_split(y_clean, 4)

    return list(zip(X_parts, y_parts))

# models used in centralized setting
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'k-NN': KNeighborsRegressor(n_neighbors=10)
}

client_data = load_data()

for target in target_names:
    print(f"\n=== Federated Round: Target = {target} ===")
    aggregated_results = {model_name: {'mse': [], 'r2': [], 'acc': []} for model_name in models}

    for client_id, (X_client, y_client) in enumerate(client_data):
        X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.3, random_state=42)

        for model_name, model in models.items():
            model.fit(X_train, y_train[target])
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test[target], y_pred)
            r2 = r2_score(y_test[target], y_pred)
            acc = model.score(X_test, y_test[target])

            aggregated_results[model_name]['mse'].append(mse)
            aggregated_results[model_name]['r2'].append(r2)
            aggregated_results[model_name]['acc'].append(acc)

    for model_name in models:
        avg_mse = np.mean(aggregated_results[model_name]['mse'])
        avg_r2 = np.mean(aggregated_results[model_name]['r2'])
        avg_acc = np.mean(aggregated_results[model_name]['acc'])

        print(f"{model_name} --> MSE: {avg_mse:.4f}, R²: {avg_r2:.4f}, Accuracy: {avg_acc:.4f}")