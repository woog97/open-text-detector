"""Resolve corpus papers against the scraped OpenReview catalogue without fuzzy guesses."""
import hashlib,json,re,unicodedata
from html.parser import HTMLParser
from urllib.parse import urlparse,parse_qs
from collections import defaultdict,Counter
from pathlib import Path
CATALOGUE=Path(__file__).resolve().parents[2]/'research/data/openreview_catalogue_all/papers.jsonl'
def normalized_title(title):
 return re.sub(r'\W+','',unicodedata.normalize('NFKC',title or '').casefold())
def key(p):return (str(p['conference']).casefold(),int(p['year']),normalized_title(p['title']))
class ForumLinks(HTMLParser):
 def __init__(self):super().__init__();self.ids=set()
 def handle_starttag(self,tag,attrs):
  if tag!='a':return
  href=dict(attrs).get('href','');url=urlparse(href)
  if url.hostname=='openreview.net' and url.path=='/forum':self.ids.update(parse_qs(url.query).get('id',[]))
def resolve(papers,source=None):
 index=defaultdict(set)
 with CATALOGUE.open() as f:
  for line in f:
   r=json.loads(line)
   if r.get('forum_id') and r.get('title') and r.get('year') is not None:index[key(r)].add(r['forum_id'])
 mapping={};statuses=Counter();details=[]
 for p in papers:
  candidates=index.get(key(p),set())
  direct=set()
  if source is not None:
   page=Path(source)/'sources'/(p['paper_id']+'.html')
   if page.exists():
    parser=ForumLinks();parser.feed(page.read_text());direct=parser.ids
  basis='unique_title_conference_year'
  if direct:
   candidates=candidates & direct;basis='official_proceedings_link_and_scraped_title_conference_year'
  existing=p.get('forum_id')
  if existing:
   assert candidates=={existing},'Existing forum ID disagrees with scraped catalogue'
  forum=next(iter(candidates)) if len(candidates)==1 else None
  status='matched' if forum else 'ambiguous' if candidates else 'not_matched_in_scraped_catalogue'
  mapping[p['paper_id']]=forum;statuses[status]+=1
  details.append({'paper_id':p['paper_id'],'forum_id':forum,'match_status':status,'match_basis':basis if forum else None})
 return mapping,{'catalogue_sha256':hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),'catalogue_source':'Local scraped OpenReview submission catalogue','match_rule':'Unique forum ID for exact normalized title plus conference and year; normalization uses NFKC, casefold and removal of non-word characters. Official proceedings forum links disambiguate duplicates and must agree when present. No fuzzy or cross-venue/year matches.','null_semantics':'No unambiguous corresponding entry in the scraped catalogue; does not prove that the paper has no OpenReview forum.','counts':dict(statuses),'papers':details}
