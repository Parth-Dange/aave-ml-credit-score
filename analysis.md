# ML-Based Wallet Scoring — Analysis & Insights

## Score Distribution

- The majority of wallets score between 200–800.
- 0–100: Frequent liquidations, failed repayment.
- 900–1000: Regular repayments, no liquidations.

## Behavioral Patterns by Score

| Score Range | Observed Behavior                                             |
| ----------- | ------------------------------------------------------------- |
| 0–200       | Borrowers liquidated, little/no repayment, rapid txn patterns |
| 800–1000    | Active, diversified, always repaid, never liquidated          |

## Model Performance

- See terminal output for precision/recall/f1 scores on the validation set.

## Conclusion

This ML scoring system allows DeFi protocols to gauge on-chain reputation using only transaction data, supporting fair lending/borrowing markets without trusting centralized identities.
