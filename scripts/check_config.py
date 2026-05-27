import yaml, sys
sys.path.insert(0, '.')
with open('configs/trend_phase4_v2.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
d = cfg['data']
t = cfg['training']
print(f"Config OK: max_rows={d['max_rows_per_symbol']} stride={d.get('stride',1)} use_btc={d['use_btc_beta']} seq_len={d['seq_len']}")
print(f"Models: {list(cfg['models'].keys())}")
for k,v in cfg['models'].items():
    print(f"  {k}: {v}")
