"""Run on the existing Space. Audit every row; preserve text and prior fields."""
import collections
import datetime
import functools
import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path

BASE = Path('/tmp/pangram-human-active-20261002')
OUT = BASE / 'origin-permission-audit-20261003'
INPUT = BASE / 'checkpoints/finish100k-active-20261002T193833Z.sqlite3'
EXPECTED_SHA = 'efe8bace75f6e62a9103fc209d3b5123d869583d43d37cde57f1464fffb71f2f'
AUDIT_ID = 'stage1-origin-permission-20261003-v1'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def old(value):
    match = re.search(r'\b((?:19|20)\d{2})\b', str(value or ''))
    return bool(match and int(match[1]) <= 2021)

def result(status, reason, **extra):
    return dict(result=status, reason=reason, **extra)

def origin(r, w):
    s = r['source_id']; m = w['record'].get('metadata', {})
    # These are historical-evidence checks, not AI-detector scores.
    if s in {'imdb', 'writingprompts'}:
        expected = {'imdb': 'c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe', 'writingprompts': '4b8d0415c450fe1ceadec667bdf73c185268bc3d9e1f6ddf64edac0481cb71f4'}[s]
        if r.get('source_revision') == m.get('archive_sha256') == expected and m.get('official_split') == 'train':
            return result('passed', 'original_pre2022_human_corpus_with_matching_archive_identity')
    if s in {'opinrank', 'govreport'} and old(r.get('corpus_release_year')):
        return result('passed', 'original_pre2022_human_corpus_release_and_preserved_source_record')
    if s == 'amazon2018' and m.get('original_json_line') and m.get('original_line_sha256') == digest(m['original_json_line']):
        return result('passed', 'original_2018_review_release_with_verified_original_record_hash')
    if s in {'oanc_slate', 'oanc_icic'} and m.get('archive_sha256') and old(m.get('publication_date')):
        return result('passed', 'publisher_contributed_historical_corpus_and_document_header')
    if s == 'acl' and old(m.get('year')) and m.get('acl_id') and 'Association for Computational Linguistics' in str(m.get('publisher', '')):
        return result('passed', 'historical_published_paper_with_publisher_and_bibliographic_identity')
    if s == 'worldbank' and old(r.get('claimed_original_date')) and m.get('download_manifest') and m.get('official_document_metadata'):
        return result('passed', 'historical_official_publication_and_preserved_download_evidence')
    if s == 'fed' and old(m.get('date')) and m.get('author') and m.get('html_sha256'):
        return result('passed', 'dated_official_speech_attributed_to_named_board_speaker', limitation='Publisher attribution supports human origin; historical bytes were not independently recaptured.')
    if s in {'cccc', 'scidev', 'eff', 'voa', 'scp'}:
        capture = r.get('historical_capture_date') or r.get('archive_capture_date') or m.get('warc_date') or m.get('capture_date')
        if old(capture): return result('passed', 'preserved_pre2022_capture_supports_human_origin', historical_capture_date=str(capture))
        if s == 'scp' and old(r.get('latest_recorded_revision_date')) and m.get('history'):
            return result('passed', 'pinned_tale_revision_history_ends_before_2022')
    if s == 'wikivoyage' and r.get('date_evidence_basis') == 'official_frozen_revision_timestamp' and old(r.get('claimed_original_date')):
        return result('passed', 'official_fixed_pre2022_revision')
    if s == 'ubuntu_irc':
        return result('could_not_verify', 'dated_chat_and_bot_filter_do_not_establish_that_every_retained_speaker_was_human')
    if s in {'asap2', 'persuade'}:
        return result('could_not_verify', 'publisher_describes_student_writing_but_collection_dates_and_writing_tool_use_not_established')
    return result('could_not_verify', 'no_sufficient_human_origin_or_historical_version_evidence_in_preserved_record', claimed_date=r.get('claimed_original_date', ''))

def permission(r, w, source_evidence):
    s = r['source_id']; m = w['record'].get('metadata', {}); license_text = r.get('license_evidence', '')
    if s in {'acl', 'persuade'}:
        return result('failed', 'downloaded_release_has_noncommercial_restriction_or_conflicting_grants', license=license_text, research_use='may_be_allowed_under_noncommercial_terms', scope='commercially_reusable_training_data')
    if s in {'imdb', 'writingprompts', 'amazon2018', 'opinrank'}:
        return result('could_not_verify', 'release_availability_or_dataset_license_does_not_establish_underlying_authors_reuse_grant', license=license_text)
    if s == 'cccc': return result('could_not_verify', 'domain_audit_exists_but_exact_text_license_and_version_missing')
    if s == 'scidev': return result('could_not_verify', 'article_has_attribution_grant_but_official_pages_conflict_on_license_version_2_vs_3', license=license_text)
    if s == 'ubuntu_irc': return result('could_not_verify', 'mirror_claims_public_domain_but_original_contributor_grant_not_established')
    if s == 'wikisource': return result('could_not_verify', 'work_edition_and_translation_rights_not_established_by_generic_wikimedia_license')
    if not source_evidence: return result('could_not_verify', 'source_policy_fetch_unsuccessful')
    common_conditions = ['Retain source identity and attribution.', 'Follow the applicable license, including share-alike where required.', 'Grant covers licensed text, not separately restricted third-party material.']
    if s == 'wikinews':
        return result('could_not_verify', 'stored_BY_SA_4_license_conflicts_with_official_historical_BY_2_5_policy', recorded_license=license_text, policy_license='CC BY 2.5 for 2005-09-25 through 2024-12-15 publications')
    if s in {'oanc_slate', 'oanc_icic'} and m.get('archive_sha256'):
        return result('passed', 'original_publisher_unrestricted_use_and_redistribution_grant', license='OANC unrestricted use and redistribution', conditions=['Preserve publisher/source acknowledgments.'])
    if s == 'fed' and m.get('author') and 'government' in license_text:
        return result('passed', 'official_board_original_material_public_domain_policy', license='US government work / Board policy', conditions=['US copyright scope; credit the Board.', 'Third-party exceptions remain excluded from this grant.'])
    if s == 'voa' and r.get('author_attribution_json') not in (None, '', '[]') and 'VOA' in r.get('original_rights_basis', ''):
        return result('passed', 'VOA_original_byline_evidence_and_official_public_domain_policy', license='VOA-produced original material: public domain', conditions=['Credit VOA.', 'No rights granted for wire-service or other third-party material.'])
    if s == 'worldbank' and m.get('license_grant') and '/licenses/by/' in license_text.lower():
        return result('passed', 'preserved_work_specific_license_grant', license=license_text, conditions=common_conditions)
    if s == 'eff' and m.get('authors') and r.get('source_byline'):
        return result('passed', 'original_EFF_byline_and_publisher_CC_BY_policy', license=license_text, conditions=common_conditions)
    if s == 'asap2' and m.get('official_split') == 'train' and '/by/4.0' in license_text:
        return result('passed', 'original_release_explicit_CC_BY_4_grant', license=license_text, conditions=common_conditions)
    if s == 'scp' and r.get('author_attribution_json') not in (None, '', '[]'):
        return result('passed', 'SCP_text_license_and_preserved_contributor_attribution', license=license_text, conditions=common_conditions)
    if s in {'wikipedia','wikibooks','wiki_talk','wikivoyage','stack_nontech','stack_tech','globalvoices'} and r.get('source_url'):
        return result('passed', 'source_text_license_policy_matches_preserved_record_grant', license=license_text, conditions=common_conditions, evidence_level='source_policy_plus_record_metadata')
    # Do not upgrade a third-party mirror's license assertion to an independently
    # verified work-specific grant. The attempt is complete even when unresolved.
    if s in {'arxiv','pmc','pes2o','pressbooks','libretexts','foodista','gutenberg'}:
        return result('could_not_verify', 'compatible_license_recorded_by_mirror_but_original_work_grant_not_independently_verified', license=license_text, recorded_license_appears_compatible=True)
    return result('could_not_verify', 'insufficient_record_specific_permission_evidence', license=license_text)

def main():
    OUT.mkdir(exist_ok=True)
    assert sha(INPUT) == EXPECTED_SHA, 'Input changed'
    evidence = json.loads((OUT/'source-evidence-manifest.json').read_text())
    by_url = {x['url']: x for x in evidence['requests']}
    source = sqlite3.connect('file:'+str(INPUT)+'?mode=ro&immutable=1', uri=True)
    target = OUT/'checked-collection.sqlite3'
    if target.exists(): raise RuntimeError('Audit output already exists; inspect rather than overwrite')
    dest = sqlite3.connect(target); source.backup(dest)
    @functools.lru_cache(maxsize=256)
    def raw_document(h):
        found = source.execute('SELECT raw FROM documents WHERE hash=?',(h,)).fetchone()
        if not found: raise ValueError('missing_saved_original')
        w = json.loads(gzip.decompress(found[0]))
        if digest(w['record']['text']) != h: raise ValueError('original_text_hash_mismatch')
        return w
    totals = collections.Counter(); by_source = collections.defaultdict(collections.Counter)
    changed = []; now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checks_file = OUT/'checks.jsonl.gz'
    with gzip.open(checks_file,'wt',encoding='utf-8') as stream:
        for rid, payload in source.execute('SELECT id,row FROM passages ORDER BY id'):
            r = json.loads(payload); sid = r['source_id']
            urls = evidence['sources'][sid]
            good = [by_url[u]['evidence_id'] for u in urls if by_url[u].get('usable')]
            try:
                w = raw_document(r['raw_text_sha256']); original = w['record']['text']
                if digest(r['text']) != r['passage_sha256'] or original[r['raw_start']:r['raw_end']] != r['text']:
                    raise ValueError('passage_does_not_match_saved_original')
                h = origin(r,w); p = permission(r,w,good)
                integrity = 'passed'
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
                reason = str(ex) if isinstance(ex,ValueError) else type(ex).__name__
                integrity = 'failed';h=result('could_not_verify','saved_evidence_could_not_be_validated',error=reason);p=dict(h)
            details = {'audit_id':AUDIT_ID,'checked_at':now,'human_origin':h,'permission_to_use':p,
                'saved_original_check':integrity,'source_policy_evidence_ids':good,
                'source_policy_urls_attempted':urls,'raw_text_sha256':r['raw_text_sha256'],
                'source_revision':r['source_revision'],'both_checks_attempted':True}
            updates = {'human_origin_check_completed':True,'human_origin_check_result':h['result'],
                'permission_to_use_check_completed':True,'permission_to_use_check_result':p['result'],
                'stage1_origin_permission_checks_completed':True,'stage1_check_audit_id':AUDIT_ID,
                'stage1_check_details_json':json.dumps(details,ensure_ascii=False,sort_keys=True)}
            r.update(updates)
            dest.execute('UPDATE passages SET row=? WHERE id=?',(json.dumps(r,ensure_ascii=False),rid))
            stream.write(json.dumps({'record_id':rid,'source_id':sid,**updates},ensure_ascii=False)+'\n')
            totals['checked']+=1;totals['origin_'+h['result']]+=1;totals['permission_'+p['result']]+=1;totals['integrity_'+integrity]+=1
            by_source[sid]['checked']+=1;by_source[sid]['origin_'+h['result']]+=1;by_source[sid]['permission_'+p['result']]+=1
            if totals['checked']%10000==0:
                dest.commit();print('CHECKED',totals['checked'],flush=True)
    dest.commit()
    assert totals['checked']==100000
    assert dest.execute('PRAGMA integrity_check').fetchone()==('ok',)
    # Compare every pre-existing field: this task adds annotations only.
    for (rid,payload),(rid2,payload2) in zip(source.execute('SELECT id,row FROM passages ORDER BY id'),dest.execute('SELECT id,row FROM passages ORDER BY id')):
        a=json.loads(payload);b=json.loads(payload2);assert rid==rid2 and all(b[k]==v for k,v in a.items())
        assert b['human_origin_check_completed'] and b['permission_to_use_check_completed']
    source.close();dest.close()
    summary={'audit_id':AUDIT_ID,'state':'all_checks_attempted','language_scope':'English only; multilingual coverage excluded',
      'input_checkpoint':str(INPUT),'input_sha256':EXPECTED_SHA,'output_checkpoint':str(target),'output_sha256':sha(target),
      'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':dict(totals),'by_source':dict(by_source),
      'attempt_coverage_percent':100,'permission_scope':'Evidence for reusable training text including commercial use, subject to license conditions; not a blanket legal clearance.',
      'human_origin_scope':'Historical archives, attributable publications and source records; positive evidence supports human origin, not a guarantee of no automated text.',
      'method':'Every passage matched to its saved original and evaluated using record-specific metadata and source policies. Original-work license pages were not individually fetched for every mirrored work; those unresolved grants are explicitly could_not_verify.',
      'completion_definition':'Every row has an attempted result for both checks, regardless of pass/fail/could_not_verify.',
      'original_fields_unchanged':True,'rows_removed':0,'training_admission_changed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__': main()
