# import pandas as pd
# import numpy as np
# import joblib  # <-- Import joblib for saving the scaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# def main():
#     # --- 1. DATA PREPARATION ---
#     print("Step 1: Preparing data...")
#     df = pd.read_csv('FINAL DATA - Sheet1.csv')
#     df.dropna(inplace=True)

#     NUM_BINS = 50
#     grouping_keys = ['frame_id', 'cluster_id']
#     pair_classification = df.groupby(grouping_keys)['classification'].apply(lambda x: x.mode().iloc[0])

#     min_z = df['z_coordinate'].min()
#     max_z = df['z_coordinate'].max()
#     bin_edges = np.linspace(min_z, max_z, NUM_BINS + 1)

#     df['bin'] = pd.cut(df['z_coordinate'], bins=bin_edges, labels=False, include_lowest=True)
#     grouped = df.groupby(grouping_keys + ['bin'])['normalized_intensity'].mean()
#     processed_df = grouped.unstack(level='bin', fill_value=0) # Changed padding to 0
#     processed_df['classification'] = pair_classification

#     X = processed_df.drop('classification', axis=1).values
#     y = processed_df['classification'].values

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     n_features = X_train.shape[1]
#     print(f"Data preparation complete. Number of features: {n_features}\n")

#     # --- 2. BUILD AND TRAIN THE NEURAL NETWORK ---
#     print("Step 2: Building and training the neural network...")
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(n_features,)),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
    
#     model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
#     print("Model training finished.\n")

#     # --- 3. EVALUATE PERFORMANCE ---
#     print("Step 3: Evaluating model performance...")
#     y_pred_proba = model.predict(X_test)
#     y_pred = (y_pred_proba > 0.5).astype(int)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"\nTest Accuracy: {accuracy:.4f}\n")

#     # === 4. SAVE ALL NECESSARY FILES FOR PREDICTION ===
#     print("Step 4: Saving model and preprocessing tools...")
    
#     # Save the trained model in H5 format
#     model.save("lidar_classifier.h5")
    
#     # Save the scaler object
#     joblib.dump(scaler, 'scaler.joblib')

#     # Save the bin edges numpy array
#     np.save('bin_edges.npy', bin_edges)
    
#     print("Successfully saved 'lidar_classifier.h5', 'scaler.joblib', and 'bin_edges.npy'.")

# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def main():
    # --- 1. DATA PREPARATION ---
    print("Step 1: Preparing data...")
    try:
        df = pd.read_csv('FINAL DATA - Sheet1.csv')
        print(f"Successfully loaded 'FINAL DATA - Sheet1.csv'. Shape: {df.shape}")
    except FileNotFoundError:
        print("\n[ERROR] 'FINAL DATA - Sheet1.csv' was not found. Please place it in the same directory as this script.\n")
        return

    df.dropna(inplace=True)
    if df.empty:
        print("\n[ERROR] DataFrame is empty after dropping missing values. Please check your CSV file.\n")
        return

    NUM_BINS = 50
    grouping_keys = ['frame_id', 'cluster_id']
    pair_classification = df.groupby(grouping_keys)['classification'].apply(lambda x: x.mode().iloc[0])

    min_z = df['z_coordinate'].min()
    max_z = df['z_coordinate'].max()
    bin_edges = np.linspace(min_z, max_z, NUM_BINS + 1)

    df['bin'] = pd.cut(df['z_coordinate'], bins=bin_edges, labels=False, include_lowest=True)
    grouped = df.groupby(grouping_keys + ['bin'])['normalized_intensity'].mean()
    processed_df = grouped.unstack(level='bin', fill_value=0)
    processed_df['classification'] = pair_classification

    # Get the feature DataFrame to save its column schema
    X_df = processed_df.drop('classification', axis=1)
    X = X_df.values
    y = processed_df['classification'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    print(f"Data preparation complete. Model will be trained on {n_features} features.\n")

    # --- 2. BUILD AND TRAIN THE NEURAL NETWORK ---
    print("Step 2: Building and training the neural network...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    print("Model training finished.\n")

    # --- 3. EVALUATE PERFORMANCE ---
    print("Step 3: Evaluating model performance...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}\n")

    # --- 4. SAVE ALL NECESSARY FILES FOR DEPLOYMENT ---
    print("Step 4: Saving model and preprocessing tools...")
    
    model.save("lidar_classifier.h5")
    joblib.dump(scaler, 'scaler.joblib')
    np.save('bin_edges.npy', bin_edges)
    # Save the exact list of feature columns the model was trained on
    np.save('feature_columns.npy', X_df.columns.astype(np.int64))
    
    print("Successfully saved all 4 required files for deployment.")

if __name__ == '__main__':
    main()