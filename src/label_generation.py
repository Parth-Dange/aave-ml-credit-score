# src/label_generation.py

import pandas as pd

def generate_labels(df, drop_uncertain=True):
    labels = []
    for _, row in df.iterrows():
        repay_ratio = row["repay_borrow_ratio"]
        liquidation_count = row["liquidation_count"]

        if repay_ratio >= 0.9 and liquidation_count == 0:
            labels.append(1)
        elif repay_ratio < 0.2 or liquidation_count > 0:
            labels.append(0)
        else:
            labels.append(-1)  # uncertain

    df["label"] = labels

    if drop_uncertain:
        df = df[df["label"] != -1]

    return df

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        exit(1)

    df = pd.read_csv(args.input)
    df_labeled = generate_labels(df)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_labeled.to_csv(args.output, index=False)

    print(f"✅ Labeled features saved to {args.output}")
