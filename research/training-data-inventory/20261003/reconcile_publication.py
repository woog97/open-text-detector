"""Apply verified post-audit publication receipts without changing census estimates."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[3]
P=R/'research/synthetic-mirrors/publication-20261003'
O=Path(__file__).parent

def reconcile():
 if not (P/'completion.json').exists():return
 completed=json.loads((P/'completion.json').read_text());assert completed['state']=='verified'
 m=json.loads((P/'mirrors-receipt.json').read_text());p=json.loads((P/'papers-receipt.json').read_text())
 assert m['state']==p['state']=='verified' and m['rows']==28120 and p['rows']==14561
 url='https://huggingface.co/datasets/'+m['repo_id']
 status='Recovered byte-for-byte and published privately; final family/split preparation remains'
 d=json.loads((O/'inventory.json').read_text())
 s=next(x for x in d['sources'] if x['id']=='mirrors28120')
 s.update(status=status,locations=[url,m['space_path'],'/data/workspace/dataset-publication-20261003/recovered-filter', 'research/synthetic-mirrors/publication-20261003/mirrors-receipt.json'],revision=m['revision'],notes=['1,880 rejected; raw 30k and original rejection evidence preserved.','Accepted/rejected/decision/protocol exports reproduced byte-for-byte against original recorded SHA256 values.','Separate linked release: source_record_id joins human record_id; authorship, label and model columns distinguish synthetic records.','Both release files and remote readback hashes verified; family-grouped splits remain unassigned.'])
 s=next(x for x in d['sources'] if x['id']=='paper_archive')
 if p['space_path'] not in s['locations']:s['locations'].append(p['space_path'])
 s['space_copy_receipt']='research/synthetic-mirrors/publication-20261003/papers-receipt.json'
 d['limitations']=[x for x in d['limitations'] if not x.startswith('Filtered synthetic export absent')]
 d['post_audit_publication']={'mirrors':m,'paper_space_copy':p,'completed_at':completed['at']}
 if not any(x['id']==m['repo_id'] for x in d['hf_repositories']):d['hf_repositories'].append({'id':m['repo_id'],'revision':m['revision'],'private':True})
 (O/'inventory.json').write_text(json.dumps(d,indent=2)+'\n')
 f=R/'research/TRAINING_DATA_INVENTORY.md';text=f.read_text()
 text=text.replace('It does not launch generation, train models, change splits, or upload datasets.','The initial audit did not change data. The subsequently requested mirror publication and paper transfer are recorded below; no generation, training or split changes were made.')
 text=text.replace('Recorded retained subset; temporary export absent; no durable filtered export found',status)
 start=text.index('**Recovery issue:**') if '**Recovery issue:**' in text else -1
 if start>=0:
  end=text.index('\n\n',start)
  text=text[:start]+f'**Recovery completed:** the old temporary working folders were absent. All 12 raw archives were restored and the frozen filter reproduced the accepted, rejected, decision and protocol files byte-for-byte. The 28,120 accepted mirrors are now published privately at [{m["repo_id"]}]({url}), revision `{m["revision"]}`, and stored at `{m["space_path"]}`. Exact original exports and checks remain at `/data/workspace/dataset-publication-20261003/recovered-filter`. The private paper archive has been copied to `{p["space_path"]}` and all 14,561 rows verified against the pinned source revision `{p["revision"]}`. These transfers followed the inventory and were explicitly requested by the user.'+text[end:]
 text=text.replace('| Accepted mirrors 28,120 | Historical temporary export absent | Main future upload candidate after recovery/reproduction; no upload launched |',f'| Accepted mirrors 28,120 | Private `{m["repo_id"]}` and persistent Space copy | Published; all file hashes and row count verified by remote readback |\n| Paper archive 14,561 | `{p["space_path"]}` | Copied from private `woog/open-text-detector-paper-text`; pinned files and rows verified |')
 text=text.replace('Upload action in this audit','Audit and subsequent requested transfers')
 text=text.replace('- Reproduce/check filtered output before training or publishing it; accepted-token count not inferred from raw average.','- Recovery and publication verified; group mirrors with their human parents before assigning final splits. Accepted-token count is not inferred from the raw average.')
 marker='### Luna mirrors accepted by automated checks\n'
 start=text.index(marker);end=text.index('\n### ',start+len(marker))
 text=text[:start]+marker+'\n'+status+'.\n\n'+f'- [{m["repo_id"]}]({url}); revision `{m["revision"]}`.\n- Persistent Space files: `{m["space_path"]}`.\n- Original accepted/rejected evidence: `/data/workspace/dataset-publication-20261003/recovered-filter`.\n- Explicit `authorship=synthetic`, `label=1`, and model metadata; join `source_record_id` to the human corpus.\n- All 28,120 records and release checksums verified; no new model calls. Final family-grouped splits remain unassigned.\n'+text[end:]
 text=text.replace('Live checks were read-only except for documentation and the requested project-rule update.','Initial inventory checks were read-only except for documentation and the requested project-rule update. Subsequent authorized transfers are supported by [publication and copy receipts](synthetic-mirrors/publication-20261003/completion.json).')
 attachment=P/'dataset-attachments.json'
 if attachment.exists():
  a=json.loads(attachment.read_text())
  if a['state']=='verified':
   d['dataset_attachments']=a
   for source_id,mount in [('mirrors28120','/datasets/synthetic-mirrors-luna-28120'),('paper_archive','/datasets/pangram-paper-text')]:
    source=next(x for x in d['sources'] if x['id']==source_id)
    source['space_dataset_mount']=mount
   (O/'inventory.json').write_text(json.dumps(d,indent=2)+'\n')
   note='**Named dataset attachments:** both repositories are attached directly to `open-text-detector/training` as read-only dataset volumes: `open-text-detector/synthetic-mirrors-luna-28120` at `/datasets/synthetic-mirrors-luna-28120` (28,120 rows) and `woog/open-text-detector-paper-text` at `/datasets/pangram-paper-text` (14,561 rows). Both mounted Parquet files were opened and row counts verified after activation. These are the canonical Space access paths; bucket copies remain as preserved recovery material. [Attachment verification](synthetic-mirrors/publication-20261003/dataset-attachments.json).'
   if '**Named dataset attachments:**' not in text:text=text.replace('## Storage and upload status\n','## Storage and upload status\n\n'+note+'\n')
 org_receipt=P/'paper-org-receipt.json'
 if org_receipt.exists():
  org=json.loads(org_receipt.read_text());assert org['state']=='verified'
  source=next(x for x in d['sources'] if x['id']=='paper_archive')
  source['organization_release']=org
  url='https://huggingface.co/datasets/'+org['repo']
  if url not in source['locations']:source['locations'].append(url)
  d['post_audit_publication']['paper_organization_release']=org
  (O/'inventory.json').write_text(json.dumps(d,indent=2)+'\n')
  text=text.replace('and `woog/open-text-detector-paper-text` at `/datasets/pangram-paper-text`','and `open-text-detector/paper-text` at `/datasets/pangram-paper-text`')
  note='**Paper organization release:** [open-text-detector/paper-text](https://huggingface.co/datasets/open-text-detector/paper-text) is private and accessible to organization members, with no restricted resource group. The [interactive sample table](https://huggingface.co/datasets/open-text-detector/paper-text/viewer/default/train) was verified at the first and last rows: 14,561 records and all 12 columns. Original `woog/open-text-detector-paper-text` remains private and unchanged. The Space attachment now uses the organization release.'
  if '**Paper organization release:**' not in text:text=text.replace('## Storage and upload status\n','## Storage and upload status\n\n'+note+'\n')
 f.write_text(text)
if __name__=='__main__':reconcile()
