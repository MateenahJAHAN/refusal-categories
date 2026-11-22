"""Unzip both deliverables, verify every reader record, fields, ordering and all blind forms."""
import csv, io, json, re, zipfile
from html.parser import HTMLParser
from build_packets import HERE, PRIVATE, load, read_rows, sha
class BankScript(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=False);self.data='';self.active=False
    def handle_starttag(self,tag,attrs):self.active=tag=='script' and dict(attrs).get('id')=='records'
    def handle_data(self,s):
        if self.active:self.data+=s
    def handle_endtag(self,tag):
        if tag=='script':self.active=False

def main():
    bank={r['id']:r['prompt'] for r in load(HERE/'source/bank.json')};key=load(PRIVATE/'blind_key.json');idx={r['blind_id']:r for r in key};assert len(idx)==182
    direct={m:{r['id']:r['response'] for r in read_rows(PRIVATE/f'replies_{m}.jsonl')} for m in ['qwen','smol']};ordered={};complete={}; terms=re.compile(r'\b(?:qwen(?:2(?:\.5)?)?|smollm(?:2)?|hugging[ -]?face(?:tb)?|alibaba|coconot|xstest)\b',re.I); mentions=[]
    for who in ['A','B']:
        p=HERE/f'reviewers/Reviewer_{who}.zip'
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            assert set(z.namelist())=={'REVIEW.html',f'reviewer_{who}.csv','RUBRIC.txt','ATTRIBUTION_AFTER_RATING/ATTRIBUTION.txt','ATTRIBUTION_AFTER_RATING/LICENSE.txt'}
            assert not any(terms.search(n) for n in z.namelist());html=z.read('REVIEW.html').decode();assert '%%' not in html;ext=BankScript();ext.feed(html);records=json.loads(ext.data)
            assert not terms.search(html.replace(ext.data,''));assert not terms.search(z.read('RUBRIC.txt').decode());assert not terms.search(z.read(f'reviewer_{who}.csv').decode());assert all(k not in html for k in ['logp_decline','top_text','help_label','generated_token_ids','model.safetensors'])
            reader=csv.DictReader(io.StringIO(z.read(f'reviewer_{who}.csv').decode()));assert reader.fieldnames==['blind_id','code','notes'];forms=list(reader);order=[r['blind_id'] for r in records];assert len(order)==len(set(order))==len(idx) and set(order)==set(idx) and order==[f['blind_id'] for f in forms] and all(f['code']==f['notes']=='' for f in forms)
            for r in records:
                assert set(r)=={'blind_id','prompt','reply'} and re.fullmatch(r'R-[0-9A-F]{12}',r['blind_id']);ks=idx[r['blind_id']];assert r['prompt']==bank[ks['id']] and r['reply']==direct[ks['model']][ks['id']]
                if who=='A':
                    for field in ['prompt','reply']:
                        for match in terms.finditer(r[field]):mentions.append(dict(blind_id=r['blind_id'],field=field,term=match.group(0)))
            ordered[who]=order;complete[who]={'n':len(order),'exact_matches':len(order),'empty_labels':len(forms),'sha256':sha(p)}
    assert ordered['A']!=ordered['B'];same=sum(u==v for u,v in zip(ordered['A'],ordered['B']));assert same<10
    out={'raw_to_reader_exact':complete,'matching_positions':same,'permutations_distinct':True,'identity_or_dataset_string_hits_in_reader_material':mentions,'attribution':'Source attribution and license are isolated in ATTRIBUTION_AFTER_RATING. External writing style can still reveal a family.'};(PRIVATE/'packet_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
