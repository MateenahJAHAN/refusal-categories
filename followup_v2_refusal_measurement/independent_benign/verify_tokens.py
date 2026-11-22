"""Replay the decoding of all saved tokens against the two immutable tokenizer snapshots."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from build_packets import HERE, PRIVATE, load, read_rows, sha

def main():
    report={};bank={r['id']:r['prompt'] for r in load(HERE/'source/bank.json')}
    for key in ['qwen','smol']:
        meta=load(PRIVATE/f'run_{key}.json');path=Path.home()/'.cache/huggingface/hub'/('models--'+meta['model'].replace('/','--'))/'snapshots'/meta['revision'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True);ends=load(path/'generation_config.json')['eos_token_id'];ends=set(ends if isinstance(ends,list) else [ends]);records=read_rows(PRIVATE/f'replies_{key}.jsonl')
        for r in records:
            seq=r['generated_token_ids'];assert tok.decode(seq,skip_special_tokens=True)==r['response'];assert len(seq)==r['answer_tokens']<=2048 and not ends.intersection(seq)
            if r['finish']=='stop':assert r['stop_id'] in ends
            elif r['finish']=='length':assert r['stop_id'] is None and len(seq)==2048
            else:raise ValueError('Unknown finish')
            assert r['initial_cap']==(r['blocks']>1)
            prompt=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':bank[r['id']]}],tokenize=True,add_generation_prompt=True,return_dict=False);assert len(prompt)==r['input_tokens']
        assert len(records)==len(bank)==91;report[key]={'decoded_exact':len(records),'input_lengths_verified':len(records),'stop_tokens':sorted(ends)}
    (PRIVATE/'token_validation.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
