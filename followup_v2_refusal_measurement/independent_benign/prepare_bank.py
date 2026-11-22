"""Rebuild the complete official stratum with byte-level source anchoring."""
from pathlib import Path
import hashlib, json, re, unicodedata
from difflib import SequenceMatcher
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REV = '2cbe16aabf9069f17e48c8daad8aeabc29469eb7'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def form(s): return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
rows = pq.read_table(HERE/'source/test-00000-of-00001.parquet').to_pylist()
cohort = [dict(id=r['id'],group='coconot',prompt=r['prompt']) for r in rows if r['subcategory']=='false presuppositions']
old = json.loads((HERE.parent/'data/items.json').read_text())
assert len(rows)==379 and len(cohort)==91 and len({r['id'] for r in cohort})==91 and len(old)==346
assert len({form(r['prompt']) for r in cohort})==91
assert not {form(r['prompt']) for r in cohort}.intersection(form(r['prompt']) for r in old)
# Maximum within each pair is a diagnostic, not a scientific threshold of independence.
pairs=[]
for r in cohort:
    s=form(r['prompt']); value, key=max((SequenceMatcher(None,s,form(t['prompt']),autojunk=False).ratio(),t['id']) for t in old)
    pairs.append(dict(id=r['id'],nearest_prior_id=key,similarity=value))
(HERE/'source/bank.json').write_text(json.dumps(cohort,indent=2,ensure_ascii=False)+'\n')
(HERE/'source/source.json').write_text(json.dumps(dict(repository='allenai/coconot',revision=REV,config='contrast',split='test',stratum='false presuppositions',source_url=f'https://huggingface.co/datasets/allenai/coconot/resolve/{REV}/contrast/test-00000-of-00001.parquet',source_rows=len(rows),selected=len(cohort),prior_rows=len(old),canonical_exact_matches=0,canonical_duplicates=0,nearest_prior=pairs,sha256={n:sha(HERE/n) for n in ['source/test-00000-of-00001.parquet','source/README.md','source/LICENSE.md','source/bank.json','../data/items.json']}),indent=2)+'\n')
