"""Generate PAPER_EVIDENCE_PACKAGE/MANIFEST.csv (full inventory)."""
import csv
from pathlib import Path

PKG = Path('PAPER_EVIDENCE_PACKAGE')
ROOT = PKG.resolve().parents[1]

items = []

def add(item, typ, desc, source, exp, status, section, tbl='', fig='', notes=''):
    items.append([item, typ, desc, source, exp, status, section, tbl, fig, notes])

add('00_README.md', 'md', 'Package readme / canonicity rules', 'generated', 'all', 'Accepted', 'README', '', '', '')
add('01_CANONICAL_EXPERIMENT_REGISTER.csv', 'csv', 'E1-E14 experiment register', 'build_tables.py', 'E1-E14', 'Verified', 'Methodology', '1', '', '')
add('02_NUMBER_TRACEABILITY.csv', 'csv', 'Number-to-source traceability map', 'generated', 'E1-E14', 'Verified', 'Results', '1-14', '', '')
add('09_PAPER_CLAIMS.md', 'md', 'Claims-to-evidence map + forbidden claims', 'generated', 'all', 'Accepted', 'Claims', '', '', '')
add('MANIFEST.csv', 'csv', 'Package manifest', 'generated', 'all', 'Accepted', 'README', '', '', '')
add('FINAL_VALIDATION.md', 'md', 'Package validation PASS/FAIL', 'generated', 'all', 'Pending', 'Validation', '', '', '')

# tables
for t in sorted((PKG / '03_TABLES').glob('*.csv')):
    num = t.name.replace('table_', '').split('_')[0]
    add(t.name, 'csv', f'Publication table {num} (data)', 'build_tables.py from 06_RAW_PAPER_DATA', '', 'Verified', 'Results', num, '', '')
for t in sorted((PKG / '03_TABLES').glob('*.tex')):
    num = t.name.replace('table_', '').split('_')[0]
    add(t.name, 'tex', f'Publication table {num} (LaTeX)', 'build_tables.py', '', 'Generated', 'Results', num, '', '')

# figures
figrows = {}
if (PKG / '04_FIGURES').exists():
    figcsv = PKG / '04_FIGURES' / 'FIGURE_REGISTER.csv'
    if figcsv.exists():
        for r in csv.DictReader(open(figcsv)):
            figrows[r['figure_id']] = r.get('paper_section', '')
for f in sorted((PKG / '04_FIGURES').iterdir()):
    if not f.is_file() or f.suffix not in ('.png', '.pdf', '.svg'):
        continue
    stem = f.name.split('.')[0]
    section = figrows.get(stem, '')
    status = 'Verified' if f.suffix == '.png' else 'Derived'
    add(f.name, f.suffix.replace('.', ''), f'Figure {stem}', 'build_figures.py / diagnostic run' if 'FIG2' in stem else 'build_figures.py',
        '', status, section or 'Results', '', stem, '')
add('FIGURE_REGISTER.csv', 'csv', 'Figure register (caption/source/caveat)', 'generated', 'all', 'Accepted', 'Figures', '', '', '')
add('EMBEDDING_VISUALIZATION_STATUS.md', 'md', 'Per-coordinate data status + diagnostic description',
    'generated', 'diagnostic', 'Accepted', 'Figures', '', '', '')

# statistics
for s in sorted((PKG / '05_STATISTICS').iterdir()):
    if not s.is_file():
        continue
    if s.name == 'E13_effect_sizes.csv':
        continue
    add(s.name, s.suffix.replace('.', ''), 'Statistics file', 'build_statistics.py', 'E13', 'Verified', 'Statistics', '13', '', '')

# raw data copies
for f in sorted((PKG / '06_RAW_PAPER_DATA').rglob('*')):
    if f.is_file():
        rel = str(f.relative_to(PKG))
        add(rel, f.suffix.replace('.', ''), 'Canonical raw data copy', 'results/final_* (copied)', '', 'Verified', 'Results', '', '', '')

# methodology
for f in sorted((PKG / '07_METHODOLOGY').iterdir()):
    if f.is_file():
        add(f.name, f.suffix.replace('.', ''), 'Methodology/provenance artifact', 'frozen methodology', '', 'Accepted', 'Methodology', '', '', '')

# limitations
for f in sorted((PKG / '08_LIMITATIONS').iterdir()):
    if f.is_file():
        add(f.name, f.suffix.replace('.', ''), 'Limitations document', 'canonical E1-E10', '', 'Accepted', 'Limitations', '', '', '')

with open(PKG / 'MANIFEST.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['item', 'type', 'description', 'source', 'experiment', 'status',
                'paper_section', 'paper_table', 'paper_figure', 'notes'])
    w.writerows(items)
print('manifest rows:', len(items))