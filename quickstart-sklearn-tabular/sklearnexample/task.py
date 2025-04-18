import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

target_names = [
    "ST58TA", "ST117TA", "ST40TA", "ST99TA", "ST32TA", "ST91TA",
    "ST60TA", "ST119TA", "ST62TA", "ST121TA"
]

def load_data():
    # Define the path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'cleaned_data.csv')
    print(f"Loading data from: {csv_path}")
    
    # Load the data from the CSV file
    df = pd.read_csv(csv_path)
    
    # Process the PTDOB column
    df['PTDOB'] = pd.to_datetime(df['PTDOB'])
    df['year'] = df['PTDOB'].dt.year
    df['month'] = df['PTDOB'].dt.month
    df['day'] = df['PTDOB'].dt.day
    df = df.drop(columns=['PTDOB'])

    # Define features and target
    X = df.drop(columns=target_names + ['RID'])
    y = df[target_names]

    # IQR Outlier Removal
    cols = ['APVOLUME', 'ABETA42', 'TAU', 'PTAU', 'PLASMAPTAU181']  # Check if these columns exist
    Q1 = X[cols].quantile(0.25)
    Q3 = X[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X[cols] < (Q1 - 1.5 * IQR)) | (X[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns)

    # Split the data into 4 parts (clients)
    X_parts = np.array_split(X_scaled, 4)
    y_parts = np.array_split(y_clean, 4)

    return list(zip(X_parts, y_parts))