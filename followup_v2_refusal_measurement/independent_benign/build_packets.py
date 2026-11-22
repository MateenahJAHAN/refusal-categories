"""Build two separate reviewer distributions. Only the subdirectory private contains the key."""
import csv, hashlib, hmac, html, io, json, re, secrets, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE=HERE/'private';OUT=HERE/'reviewers'
ATTR='Questions from CoCoNot, Faeze Brahman, Sachin Kumar and colleagues (2024), Allen Institute for AI. https://doi.org/10.52202/079017-1573\nhttps://huggingface.co/datasets/allenai/coconot/tree/2cbe16aabf9069f17e48c8daad8aeabc29469eb7\nThe source repository includes the ODC-By license and documents component terms. See the enclosed license and the linked source card. Replies were generated for this study.\n'

def load(p):return json.loads(p.read_text())
def read_rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put_csv(rows):
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=['blind_id','code','notes']);w.writeheader();w.writerows(rows);return buf.getvalue()

def main():
    bank={r['id']:r['prompt'] for r in load(HERE/'source/bank.json')}; OUT.mkdir(exist_ok=True)
    secret=PRIVATE/'blind_seed.hex'
    if not secret.exists():secret.write_text(secrets.token_hex(32)+'\n');secret.chmod(0o600)
    seed=bytes.fromhex(secret.read_text().strip()); archive=[]; key=[]; names=[]
    for model in ['qwen','smol']:
        records=read_rows(PRIVATE/f'replies_{model}.jsonl');assert len(records)==len(bank) and {r['id'] for r in records}==set(bank)
        for r in records:
            ident=r['id'];blind='R-'+hmac.new(seed,f'{model}/{ident}'.encode(),'sha256').hexdigest()[:12].upper();text=r['response']
            mentions=re.findall(r'(?i)qwen[\w.\-]*|smollm[\w.\-]*|hugging[\s-]*face|alibaba',text)
            if mentions:names.append({'blind_id':blind,'terms':mentions})
            # Brand names can be factual names in answers (e.g. Alibaba); retain verbatim and inspect each hit individually before distribution.
            archive.append(dict(blind_id=blind,prompt=bank[ident],reply=text));key.append(dict(blind_id=blind,model=model,id=ident))
    ids={r['blind_id'] for r in key};assert len(ids)==len(archive)==182
    (PRIVATE/'blind_key.json').write_text(json.dumps(key,indent=2)+'\n');(PRIVATE/'identity_term_scan.json').write_text(json.dumps(names,indent=2)+'\n')
    for label in ['A','B']:
        items=sorted(archive,key=lambda r:hmac.new(seed,(label+'/'+r['blind_id']).encode(),'sha256').digest()); doc=json.dumps(items,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026'); rubric=(HERE/'rubric.txt').read_text()
        page=(HERE/'review_template.html').read_text().replace('%%RUBRIC%%',html.escape(rubric)).replace('%%RATER%%',label).replace('%%RECORDS%%',doc)
        files={'REVIEW.html':page.encode(),f'reviewer_{label}.csv':put_csv([dict(blind_id=r['blind_id'],code='',notes='') for r in items]).encode(),'RUBRIC.txt':rubric.encode(),'ATTRIBUTION_AFTER_RATING/ATTRIBUTION.txt':ATTR.encode(),'ATTRIBUTION_AFTER_RATING/LICENSE.txt':(HERE/'source/LICENSE.md').read_bytes()}
        with zipfile.ZipFile(OUT/f'Reviewer_{label}.zip','w') as z:
            for n,data in files.items():
                info=zipfile.ZipInfo(n,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
if __name__=='__main__':main()
