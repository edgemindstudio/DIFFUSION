
import json, sys, csv
from pathlib import Path

def main(jsonl_paths, out_csv):
    rows = []
    for p in jsonl_paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            m_ro = obj['metrics']['real_only']; m_rs = obj['metrics']['real_plus_synth']
            rows.append({
                'model': obj['model'],'seed': obj['seed'],
                'fid_macro': obj['metrics'].get('fid_macro'),'cfid_macro': obj['metrics'].get('cfid_macro'),
                'js': obj['metrics'].get('js'),'kl': obj['metrics'].get('kl'),'diversity': obj['metrics'].get('diversity'),
                'acc_R': m_ro['accuracy'],'f1_R': m_ro['macro_f1'],'balacc_R': m_ro['bal_acc'],'auprc_R': m_ro['macro_auprc'],
                'r@1%FPR_R': m_ro['recall_at_1pct_fpr'],'ece_R': m_ro['ece'],'brier_R': m_ro['brier'],
                'acc_R+S': m_rs['accuracy'],'f1_R+S': m_rs['macro_f1'],'balacc_R+S': m_rs['bal_acc'],'auprc_R+S': m_rs['macro_auprc'],
                'r@1%FPR_R+S': m_rs['recall_at_1pct_fpr'],'ece_R+S': m_rs['ece'],'brier_R+S': m_rs['brier'],
                'd_acc': obj['deltas'].get('accuracy'),'d_f1': obj['deltas'].get('macro_f1'),'d_balacc': obj['deltas'].get('bal_acc'),
                'd_auprc': obj['deltas'].get('macro_auprc'),'d_r@1%FPR': obj['deltas'].get('recall_at_1pct_fpr'),'d_ece': obj['deltas'].get('ece'),'d_brier': obj['deltas'].get('brier'),
            })
    if not rows:
        print("No data found.", file=sys.stderr); return 1
    keys = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows."); return 0

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/aggregate_results.py out.csv run1.jsonl [run2.jsonl ...]")
        sys.exit(2)
    out_csv = sys.argv[1]; jsonl_paths = sys.argv[2:]; sys.exit(main(jsonl_paths, out_csv))
