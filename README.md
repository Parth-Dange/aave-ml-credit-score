# Aave V2 ML-Based Wallet Credit Scoring

This project generates credit scores (0–1000) for each wallet based on its historical behavior on the Aave V2 protocol. We use machine learning (Random Forest Classifier) trained on engineered behavioral features with proxy labels derived from repayment and liquidation patterns.

---

## Project Structure

aave-ml-credit-score/
├── data/ # Contains user_transactions.json (not committed)
│
├── src/ # All source code
│ ├── feature_engineering.py # Extract wallet features
│ ├── label_generation.py # Generate proxy labels
│ ├── train_model.py # Train ML model
│ ├── predict_scores.py # Generate credit scores
│ └── run_pipeline.py (optional) # End-to-end runner script
│
├── output/ # Generated results (not committed)
│ ├── features.csv # Wallet features
│ ├── labeled_features.csv # Proxy-labeled data
│ ├── model.pkl # Trained model
│ └── wallet_scores_ml.csv # Final wallet → credit scores
│
├── notebooks/ # Optional notebooks for EDA and charts
│ └── eda_and_analysis.ipynb
│
├── README.md # Project instructions
├── analysis.md # Credit score summary & behavioral insights
├── requirements.txt # Python packages used
└── .gitignore # Prevents committing data/output folders

---

## Setup Instructions

### Step 1: Install Dependencies

pip install -r requirements.txt

### 1️⃣ Extract Features

From raw transaction data:

python src/feature_engineering.py --input data/user_transactions.json --output output/features.csv

### 2️⃣ Generate Labels

Create proxy labels (1 = good, 0 = risky):

python src/label_generation.py --input output/features.csv --output output/labeled_features.csv

### 3️⃣ Train Machine Learning Model

Train a Random Forest on labeled data:

python src/train_model.py --input output/labeled_features.csv --model output/model.pkl

### 4️⃣ Predict Wallet Credit Scores

Predict risk scores (0–1000):

python src/predict_scores.py --features output/features.csv --model output/model.pkl --output output/wallet_scores_ml.csv

## 📊 Output Files

| File                   | Description                            |
| ---------------------- | -------------------------------------- |
| `features.csv`         | Engineered wallet-level behavior data  |
| `labeled_features.csv` | Includes labels (0 = bad, 1 = good)    |
| `model.pkl`            | Trained ML model                       |
| `wallet_scores_ml.csv` | Final predicted credit scores (0–1000) |

## 🔬 Model & Scoring Logic

We use a Random Forest Classifier trained on:

- Deposit, borrow, repay, redeem, liquidation counts
- Ratios: repaid/borrowed, liquidations/borrows
- Total deposit and borrow values (USD)
- Active days, average transaction interval
- Number of unique assets interacted with

The final credit score is calculated as:

credit_score = round(probability_of_good_label \* 1000)

## 📦 requirements.txt

pandas
numpy
scikit-learn
joblib

## 👤 Wallet Score Output Sample

wallet,credit_score
0x00000000001accfa9cef68cf5371a23025b6d4b6,842
0x1234abcd5678ef90123456789abcdef123456789,172
...

## 📘 Notes

- No personally identifiable info is used. Wallets are scored purely on aggregated DeFi behavior.
- Labels are generated using repayment ratios and liquidation signals.
- The model is designed to be easily extensible for future datasets and protocols.
