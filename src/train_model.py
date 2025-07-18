import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(df, model_output_path):
    # Features to use (update if you change features)
    features = [
        "deposit_count", "borrow_count", "repay_count", "redeem_count",
        "liquidation_count", "total_deposit_usd", "total_borrow_usd",
        "total_repay_usd", "repay_borrow_ratio", "liquidation_borrow_ratio",
        "assets_used", "active_days", "avg_tx_interval_days"
    ]
    X = df[features]
    y = df["label"]

    # Split: 80% train, 20% test for metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)

    # Print metrics for analysis.md or README.md
    y_pred = clf.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=3))

    return clf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../output/labeled_features.csv')
    parser.add_argument('--model', type=str, default='../output/model.pkl')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train_model(df, args.model)
    print(f"Model saved to {args.model}")
