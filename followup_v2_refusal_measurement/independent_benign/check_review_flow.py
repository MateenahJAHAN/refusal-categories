"""Exercise actual reviewer pages offline in temporary Chrome profiles; leave original forms blank."""
import csv, io, json, tempfile, zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright
from build_packets import HERE, sha

def main():
    results={}
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser=p.chromium.launch(channel='chrome',headless=True,args=['--disable-component-update','--disable-background-networking'])
        for who in ['A','B']:
            d=Path(tmp)/who;d.mkdir();ctx=browser.new_context(offline=True,accept_downloads=True);page=ctx.new_page();errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
            with zipfile.ZipFile(HERE/f'reviewers/Reviewer_{who}.zip') as z:z.extractall(d)
            page.goto((d/'REVIEW.html').as_uri());data=json.loads(page.locator('#records').text_content());assert len(data)==182
            # Exercise navigation for every entry, including identical replies if any.
            for i,r in enumerate(data):
                page.select_option('#jump',index=i);assert page.locator('#code').text_content()==r['blind_id'],(i,page.locator('#jump').input_value(),page.locator('#code').text_content(),r['blind_id']);assert page.locator('#question').text_content()==r['prompt'];assert page.locator('#reply').text_content()==r['reply']
            page.select_option('#jump',index=0);page.get_by_role('radio',name='engaged',exact=True).check();note='Example, with "quotes"\nand a newline';page.locator('#notes').fill(note);page.reload();assert page.get_by_role('radio',name='engaged',exact=True).is_checked();assert page.locator('#notes').input_value()==note
            page.locator('#empty').click();assert page.locator('#jump').input_value()=='1';assert page.locator('#status').text_content()=='Item 2 of 182 · 1 coded'
            with page.expect_download() as ev:page.locator('#save').click()
            path=d/'test_only.csv';ev.value.save_as(path);returned=list(csv.DictReader(io.StringIO(path.read_text())));assert len(returned)==182 and returned[0]['code']=='engaged' and returned[0]['notes']==note and all(r['code']=='' for r in returned[1:]);page.locator('#load').set_input_files(path);page.wait_for_function("document.getElementById('msg').textContent==='Review restored.'");page.select_option('#jump',index=0);assert page.locator('#notes').input_value()==note
            page.locator('#load').set_input_files(d/f'reviewer_{who}.csv');page.wait_for_function("document.getElementById('status').textContent==='Item 1 of 182 · 0 coded'");assert not errors
            results[who]={'displayed_every_record':len(data),'browser_autosave_export_restore':'pass','browser_errors':errors,'original_zip_sha256':sha(HERE/f'reviewers/Reviewer_{who}.zip')};ctx.close()
        browser.close()
    (HERE/'private/browser_validation.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
