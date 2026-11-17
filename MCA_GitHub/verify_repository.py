from pathlib import Path
import csv
import hashlib

root = Path(__file__).resolve().parent
manifest = root / 'SHA256SUMS.csv'
rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8')))
bad = []
for row in rows:
    name = row.get('path') or row.get('file')
    target = root / name
    actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
    if actual != row['sha256']:
        bad.append(name)
print(f'Integrity: {len(rows) - len(bad)}/{len(rows)} listed files matched.')
if bad:
    print('Mismatches:', *bad, sep='\n - ')
missing = [
    f'k{k}_{objective}_seed_1_final_spectrum.csv'
    for k in (8, 16) for objective in ('raw', 'normalized')
    if not (root / 'data' / f'k{k}_{objective}_seed_1_final_spectrum.csv').is_file()
]
if missing:
    print('DFR post-hoc source spectra unavailable (expected):', *missing, sep='\n - ')
raise SystemExit(1 if bad else 0)
