import pandas as pd
import joblib
import os

def predict_scores(features_csv, model_path, output_path):
    df = pd.read_csv(features_csv)

    # Ensure these features match training
    feature_cols = [
        "deposit_count", "borrow_count", "repay_count", "redeem_count", 
        "liquidation_count", "total_deposit_usd", "total_borrow_usd",
        "total_repay_usd", "repay_borrow_ratio", "liquidation_borrow_ratio",
        "assets_used", "active_days", "avg_tx_interval_days"
    ]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    
    # Predict probability that each wallet is "good" = label 1
    X = df[feature_cols]
    prob_good = model.predict_proba(X)[:, 1]  # prob of label == 1

    df["credit_score"] = (prob_good * 1000).round().astype(int)

    result = df[["wallet", "credit_score"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"✅ Credit scores saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="../output/features.csv")
    parser.add_argument("--model", type=str, default="../output/model.pkl")
    parser.add_argument("--output", type=str, default="../output/wallet_scores_ml.csv")
    args = parser.parse_args()

    predict_scores(args.features, args.model, args.output)
