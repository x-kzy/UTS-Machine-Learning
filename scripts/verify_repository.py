"""Pemeriksaan sumber, tautan lokal, berkas hasil dan gambar paket repository."""
from pathlib import Path
import ast
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

def main():
    notebook = json.loads((ROOT/'notebooks/UTS_ML_B10_Kelompok5.ipynb').read_text(encoding='utf-8'))
    provenance = json.loads((ROOT/'config/provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'notebooks/UTS_ML_B10_Kelompok5.ipynb').read_bytes()).hexdigest() == provenance['canonical_notebook']['sha256']
    assert hashlib.sha256((ROOT/'reports/Laporan_UTS_ML_B10_Kelompok5.docx').read_bytes()).hexdigest() == provenance['report']['sha256']
    code = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    canonical = ast.parse(code).body
    exported = ast.parse((ROOT/'src/uts_ml.py').read_text(encoding='utf-8')).body
    assert [ast.dump(n,include_attributes=False) for n in exported[:len(canonical)]] == [ast.dump(n,include_attributes=False) for n in canonical], 'Kode komputasi sumber berubah'
    for file in ROOT.rglob('*.py'):
        compile(file.read_text(encoding='utf-8'),str(file),'exec')
    problems=[]; checked=0
    for file in ROOT.rglob('*.md'):
        for link in re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)',file.read_text(encoding='utf-8')):
            link=link.strip().split(' "',1)[0].strip('<>')
            if not link or link.startswith(('http:','https:','mailto:','#','data:')): continue
            target=unquote(link.split('#',1)[0])
            checked+=1
            if not (file.parent/target).exists(): problems.append((str(file.relative_to(ROOT)),target))
    assert not problems, 'Tautan lokal rusak: '+repr(problems)
    from PIL import Image
    images=list((ROOT/'docs/images').glob('*.png'))
    for path in images:
        with Image.open(path) as im: im.verify()
    assert len(images)==32
    env=json.loads((ROOT/'results/reproduced/latest/environment.json').read_text())
    assert env['scikit-learn']=='1.8.0'
    assert not (ROOT/'data/MetroPT3(AirCompressor).csv').exists(), 'Dataset lengkap tidak termasuk paket Git biasa'
    assert not (ROOT/'data/metropt+3+dataset.zip').exists()
    print(json.dumps({'status':'passed','canonical_code_cells':sum(c['cell_type']=='code' for c in notebook['cells']),
                      'local_links_checked':checked,'images_verified':len(images),'original_report_and_notebook_hashes':'matched'},indent=2))

if __name__=='__main__': main()
