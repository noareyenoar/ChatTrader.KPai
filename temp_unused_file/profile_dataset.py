import os
import pandas as pd
import numpy as np
from pathlib import Path

class DataQualityGate:
    def __init__(self, min_history=50000, purge_gap=20):
        self.min_history = min_history
        self.purge_gap = purge_gap

    def check(self, df):
        if len(df) < self.min_history:
            return False, f"Too short: {len(df)}"
        return True, "Accepted"

def main():
    data_dir = Path('data/crypto/futures/1m')
    if not data_dir.exists():
        print(f"Directory {data_dir.absolute()} does not exist.")
        return

    gate = DataQualityGate(min_history=50000, purge_gap=20)
    
    accepted_stats = []
    rejected_count = 0
    accepted_count = 0
    
    files = list(data_dir.glob('*.csv')) + list(data_dir.glob('*.parquet'))
    
    for f in files:
        try:
            if f.suffix == '.csv':
                df = pd.read_csv(f)
            else:
                df = pd.read_parquet(f)
            
            is_accepted, reason = gate.check(df)
            
            if is_accepted:
                accepted_count += 1
                stats = {'symbol': f.stem, 'rows': len(df)}
                if 'trades' in df.columns:
                    stats['trades_median'] = df['trades'].median()
                if 'quote_volume' in df.columns:
                    stats['qv_median'] = df['quote_volume'].median()
                if 'open_interest' in df.columns:
                    stats['oi_zero_frac'] = (df['open_interest'] == 0).mean()
                if 'funding_rate' in df.columns:
                    stats['fr_zero_frac'] = (df['funding_rate'] == 0).mean()
                accepted_stats.append(stats)
            else:
                rejected_count += 1
        except Exception as e:
            rejected_count += 1

    if not accepted_stats:
        print(f"Accepted: 0, Rejected: {rejected_count}")
        return

    df_s = pd.DataFrame(accepted_stats)
    print(f"Accepted: {accepted_count}, Rejected: {rejected_count}")
    print("\nTop 15:")
    print(df_s.nlargest(15, 'rows')[['symbol', 'rows']].to_string(index=False))
    print("\nBottom 10:")
    print(df_s.nsmallest(10, 'rows')[['symbol', 'rows']].to_string(index=False))
    
    print("\nMedians:")
    if 'trades_median' in df_s.columns: print(f"Trades: {df_s['trades_median'].median():.1f}")
    if 'qv_median' in df_s.columns: print(f"QuoteVol: {df_s['qv_median'].median():.1f}")
    if 'oi_zero_frac' in df_s.columns: print(f"OI_Zero: {df_s['oi_zero_frac'].median():.4f}")
    if 'fr_zero_frac' in df_s.columns: print(f"FR_Zero: {df_s['fr_zero_frac'].median():.4f}")

if __name__ == '__main__':
    main()
