import json
import pandas as pd
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_features(transactions):
    wallets = defaultdict(list)
    for tx in transactions:
        wallets[tx['userWallet']].append(tx)

    records = []
    for wallet, txs in wallets.items():
        txs_sorted = sorted(txs, key=lambda x: x['timestamp'])

        deposit_count = borrow_count = repay_count = redeem_count = liquidation_count = 0
        total_deposit_usd = total_borrow_usd = total_repay_usd = total_redeem_usd = 0

        assets = set()
        timestamps = []

        for tx in txs_sorted:
            action = tx.get('action')
            timestamp = tx.get('timestamp')
            timestamps.append(timestamp)
            action_data = tx.get('actionData', {})

            asset = action_data.get('assetSymbol', 'UNKNOWN')
            assets.add(asset)
            price = float(action_data.get('assetPriceUSD', 1))
            amount = float(action_data.get('amount', 0)) / 1e6

            usd_value = price * amount

            if action == 'deposit':
                deposit_count += 1
                total_deposit_usd += usd_value
            elif action == 'borrow':
                borrow_count += 1
                total_borrow_usd += usd_value
            elif action == 'repay':
                repay_count += 1
                total_repay_usd += usd_value
            elif action == 'redeemunderlying':
                redeem_count += 1
                total_redeem_usd += usd_value
            elif action == 'liquidationcall':
                liquidation_count += 1

        first_time = min(timestamps)
        last_time = max(timestamps)
        days_active = (last_time - first_time) / 86400 if first_time != last_time else 1
        avg_tx_interval_days = days_active / len(txs_sorted) if len(txs_sorted) > 0 else 0

        repay_borrow_ratio = total_repay_usd / total_borrow_usd if total_borrow_usd > 0 else 0
        liquidation_borrow_ratio = liquidation_count / borrow_count if borrow_count > 0 else 0

        records.append({
            "wallet": wallet,
            "deposit_count": deposit_count,
            "borrow_count": borrow_count,
            "repay_count": repay_count,
            "redeem_count": redeem_count,
            "liquidation_count": liquidation_count,
            "total_deposit_usd": total_deposit_usd,
            "total_borrow_usd": total_borrow_usd,
            "total_repay_usd": total_repay_usd,
            "repay_borrow_ratio": repay_borrow_ratio,
            "liquidation_borrow_ratio": liquidation_borrow_ratio,
            "assets_used": len(assets),
            "active_days": days_active,
            "avg_tx_interval_days": avg_tx_interval_days
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../data/user_transactions.json")
    parser.add_argument("--output", type=str, default="../output/features.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = load_json(args.input)
    df = extract_features(data)
    df.to_csv(args.output, index=False)
    print(f"Features saved to {args.output}")
