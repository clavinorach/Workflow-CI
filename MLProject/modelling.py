import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script MLP (Basic Autolog)")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="preprocess_dataset",
        help="Path to the folder containing train_preprocessed.csv and test_preprocessed.csv"
    )
    return parser.parse_args()

def load_processed_data(data_dir):
    train_p = os.path.join(data_dir, "train_preprocessed.csv")
    test_p  = os.path.join(data_dir, "test_preprocessed.csv")
    
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        print(f"[WARN] Not found in {data_dir}, trying relative path...")
        if os.path.exists("preprocess_dataset/train_preprocessed.csv"):
            train_p = "preprocess_dataset/train_preprocessed.csv"
            test_p = "preprocess_dataset/test_preprocessed.csv"
        else:
            raise FileNotFoundError(f"Data not found in {train_p} or default path.")

    print(f"[INFO] Loading data from: {train_p}")
    train_df = pd.read_csv(train_p)
    test_df  = pd.read_csv(test_p)
    target_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col]).values.astype("float32")
    y_train = train_df[target_col].values
    X_test  = test_df.drop(columns=[target_col]).values.astype("float32")
    y_test  = test_df[target_col].values
    n_classes = len(np.unique(y_train))
    return X_train, X_test, y_train, y_test, n_classes

def main():
    args = parse_args()
    
    # Unset MLFLOW_RUN_ID to avoid conflicts with parent mlflow run command
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']
    
    # Basic Criteria: Localhost & Autolog
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("sklearn_basic_autolog")
    
    # Enable Autolog (Logs params, metrics, model automatically)
    # This satisfies the requirement to remove manual logs
    mlflow.sklearn.autolog()
    
    X_train, X_test, y_train, y_test, n_classes = load_processed_data(args.data_dir)
    
    # Simple Pipeline
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=200,
            random_state=42
        ))
    ])
    
    # Train
    with mlflow.start_run():
        mlp.fit(X_train, y_train)
        
        # Evaluate
        score = mlp.score(X_test, y_test)
        print(f"Test Accuracy: {score}")

if __name__ == "__main__":
    main()
