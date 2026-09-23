"""Build run_comparison.csv from per-run results.json files. No dependencies."""
import csv, json, pathlib, sys

LEAD = ['train_images', 'epochs_run', 'best_epoch', 'phase2_minutes',
        'best_train_acc_clean', 'best_val_acc', 'gap_clean_minus_val',
        'test_accuracy', 'test_QWK', 'test_macro F1',
        'test_recall No_DR', 'test_recall Mild', 'test_recall Moderate',
        'test_recall Severe', 'test_recall Proliferative_DR',
        'idrid_accuracy', 'idrid_QWK', 'idrid_macro F1',
        'referral_threshold', 'test_referral sens @thr', 'test_referral spec @thr',
        'idrid_referral sens @thr', 'idrid_referral spec @thr', 'split_id']

src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path('.')
files = sorted(src.glob('*.json'))
if not files:
    sys.exit(f'no .json files in {src.resolve()}')

rows = []
for f in files:
    r = json.loads(f.read_text(encoding='utf-8'))
    r['run_tag'] = r.get('run_tag', f.stem)
    rows.append(r)

seen = {k for r in rows for k in r}
cols = ['run_tag'] + [c for c in LEAD if c in seen] + sorted(seen - set(LEAD) - {'run_tag'})
rows.sort(key=lambda r: r.get('test_QWK', 0), reverse=True)

out = src.parent / 'run_comparison.csv' if src.name == 'per_run' else src / 'run_comparison.csv'
with out.open('w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

show = [c for c in ['run_tag', 'train_images', 'test_accuracy', 'test_QWK', 'test_macro F1',
                    'test_recall Mild', 'test_recall Severe', 'idrid_accuracy', 'idrid_QWK'] if c in cols]
width = {c: max(len(c), *(len(f'{r.get(c, ""):.4f}' if isinstance(r.get(c), float) else str(r.get(c, ""))) for r in rows)) for c in show}
print('  '.join(c.ljust(width[c]) for c in show))
for r in rows:
    print('  '.join((f'{r[c]:.4f}' if isinstance(r.get(c), float) else str(r.get(c, ''))).ljust(width[c]) for c in show))
print('\nwritten:', out.resolve())