from pathlib import Path
import csv
import hashlib

root = Path(__file__).resolve().parent
manifest = root / 'SHA256SUMS.csv'
with manifest.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'sha256'])
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == manifest:
            continue
        if any(part in {'.git', '.venv', '__pycache__'} for part in path.relative_to(root).parts):
            continue
        writer.writerow([
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest()
        ])
print(f'Updated {manifest}')
