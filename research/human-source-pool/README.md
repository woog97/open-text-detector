# Curated human source distribution for Pangram 4

Researched October 1, 2026. This is the completed source-selection and distribution design for step 1. It is **not an assembled or validated training corpus**: no source passages have yet been admitted. The deliverables are a researched inventory, explicit sampling allocations, pinned collection metadata, and admission rules.

The recommendation is to preserve Pangram 4's nine broad category shares, then diversify sources *within* each category. Use historical text with verifiable versions as the starting human pool. Quality here means reliable provenance, faithful extraction, sufficient independent authors, and representative writing—not just polished or highly rated prose.

The paper says its human data is commercially reusable or company-owned, but does not disclose a reproducible source inventory, per-source weights, language proportions, or an unambiguous sampling unit for the category chart. We can reconstruct its broad distribution, not recover its private collection. The figures below come from Figure 2; source choices and every within-category percentage are our proposals. [Pangram 4, §2.1 and Figure 2](https://arxiv.org/html/2607.27183v1#S2.SS1).

## The proposed distribution

Start English-first. Keep multilingual expansion as a separate design until its language allocation is specified. Do not silently equate English-first with Pangram's full multilingual pool.

| Category | Paper's displayed share | Planning passages out of 100,000 | Proposed mix within the category |
|---|---:|---:|---|
| Creative writing | 22.2% | 22,222 | 45% Gutenberg / Standard Ebooks; 30% WritingPrompts*; 20% SCP tales; 5% Wikisource |
| Scientific and medical | 18.0% | 18,018 | 35% PMC licensed articles; 30% licensed arXiv full texts; 20% licensed peS2o; 15% ACL 2016–2021 |
| Reference and educational | 15.9% | 15,916 | 50% Wikipedia; 25% Pressbooks; 15% LibreTexts; 10% Wikibooks |
| Consumer reviews | 10.6% | 10,611 | 50% Amazon 2018*; 30% Yelp*; 10% IMDb*; 10% OpinRank* |
| Social, Q&A, chat | 10.4% | 10,410 | 55% nontechnical Stack Exchange; 20% technical Stack Exchange; 15% Ubuntu IRC; 10% Wikipedia discussion |
| General web and mixed | 8.2% | 8,208 | 65% dated Creative Commons Common Crawl; 10% EFF; 10% Foodista narrative prose; 15% Wikivoyage |
| News | 6.6% | 6,607 | 25% Global Voices; 25% original VOA; 15% Wikinews; 15% SciDev.Net; 20% OANC's contributed Slate files |
| Essays and academic writing | 4.9% | 4,905 | 40% ASAP 2.0; 25% ELLIPSE*; 20% BAWE*; 15% PERSUADE* |
| Professional and finance | 3.1% | 3,103 | 35% GovReport; 25% Federal Reserve Board; 25% World Bank licensed reports; 15% OANC ICIC correspondence |

**An asterisk marks restricted, conflicting, or unresolved text rights.** It is a content-fit allocation held open pending resolution, not an instruction to ingest those texts under Pangram's stated commercial-use standard. Sources without an asterisk still require document-level rights, date, quality and overlap checks. ASAP 2.0 particularly needs stronger evidence of writing dates and conditions.

The paper's displayed percentages total 99.9%. The planning allocation normalizes by 99.9 and uses largest-remainder rounding, first by category and then by source. Counts are proposed **passages**, not verified available documents or claims about Pangram's sampling unit. Track unique documents, authors and token exposure separately.

At this size, 79,779 proposed passages have an identified route through a source offering an open subset; 20,221 occupy restricted or unresolved-rights slots. Neither number is an available-record count. **Actual admitted passages: zero.** Do not remove the unresolved slots and relabel the remaining 79.8% as a matched Pangram distribution.

## Why these sources belong in each pool

### Creative writing

Use clean public-domain fiction as a stable foundation, but keep substantial contemporary amateur writing. A Gutenberg-only pool would teach a strong historical-style shortcut. Standard Ebooks is best used as an alternate transcription of the same work, not extra independent data. Select novels and short fiction deliberately; a book repository also contains nonfiction, verse and drama. [Gutenberg permissions](https://www.gutenberg.org/policy/permission.html), [Standard Ebooks](https://standardebooks.org/).

SCP is a concrete contemporary alternative with explicit share-alike terms. Prefer its narrative tales to formulaic containment entries. Its single fictional universe and horror bias justify the 20% category cap. WritingPrompts provides a broader prompt-conditioned amateur-fiction candidate, but its release README does not establish a commercial grant for the underlying Reddit stories. Keep that quota conditional. [SCP licensing](https://scp-wiki.wikidot.com/licensing-guide), [original WritingPrompts release](https://github.com/facebookresearch/fairseq/tree/main/examples/stories).

Proposed fiction strata: 35% general/literary; 25% speculative; 15% mystery/adventure; 15% relationships/coming-of-age; 10% other narrative prose. Classify stories independently of provider; do not infer genre from dataset name alone. These are coverage targets, not measured corpus availability. Contemporary professionally edited fiction remains a gap. Neither more nineteenth-century novels nor more SCP entries closes it.

### Scientific and medical

Prefer JATS or structurally reliable full text. Use PMC for biomedical prose, licensed arXiv for quantitative fields, ACL for a controlled NLP slice, and licensed peS2o to fill disciplinary gaps. The source allocation should **not** allow PMC plus biomedical peS2o to dominate twice. A proposed field allocation is 35% life sciences/medicine, 25% physical sciences/mathematics, 20% computing/engineering, 15% social/behavioral science and 5% interdisciplinary work. Reconcile this field target against source supply after deduplication.

Within papers, target 15% abstracts, 25% introductions/background, 20% methods, 20% results and 20% discussion/conclusions. Preserve both formula-adjacent and nontechnical prose; remove isolated equations and tables from the written-prose pool. Retain ordinary methods paragraphs even when repetitive: legitimate human formulaic writing is important for false-positive control.

PMC requires article-level license checks; arXiv full text has varying licenses. ACL's own copyright page distinguishes pre-2016 NC material from later CC BY material. This changes which new papers we should admit; it does not authorize rewriting existing project datasets or their historical results. [PMC OA subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/), [arXiv licenses](https://info.arxiv.org/help/license/index.html), [ACL copyright](https://aclanthology.org/faq/copyright/).

### Reference and educational

Wikipedia supplies broad factual exposition; textbooks supply longer explanations, examples and teaching style. Use exact historical revisions and book editions. Allocate subjects across humanities, social sciences, STEM, health and practical/vocational learning. Do not choose solely by an educational-quality score, which would underrepresent simple, informal and imperfect human writing.

Pressbooks provides concrete license-filtered discovery. Catalog membership does not make every chapter equally reusable; the same book can also appear in LibreTexts or other OER catalogs. OpenStax remains a useful **edition-specific backup**, but its current official guidance lists NC-SA terms. An older permissively licensed edition needs its own preserved text and notice. [Pressbooks Directory](https://pressbooks.directory/), [OpenStax current guidance](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks).

### Consumer reviews

The content recommendation is Amazon for products, Yelp for services, IMDb for long entertainment reviews and OpinRank for travel/automotive experiences. The 2018 Amazon release is preferable to the 2023 release for this purpose because its collection predates the present LLM era; it also has useful product, author, rating and date fields. Avoid exclusively using five-core subsets, which change the reviewer and product population. [Amazon 2018 release](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html).

Proposed rating strata: 20% one-star, 15% two-star, 20% three-star, 20% four-star and 25% five-star equivalents. This intentionally covers rare middle ratings and is not a claim about the natural review distribution. Preserve complaints, neutral evaluations, poor grammar and short conversational reviews. Remove seller descriptions, repeated review templates and duplicated product variants.

This is the largest unresolved acquisition issue. Yelp's academic agreement and Amazon MARC's academic-only license do not provide the unrestricted source Pangram describes. OpinRank's UCI CC BY label also does not itself establish the upstream reviewers' grants. The practical route is a suitable rights-holder release or a licensed historical review archive, not relabeling travel guides as reviews. No outside party has been contacted. [Yelp agreement](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf), [Amazon MARC license](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt), [OpinRank](https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset).

### Social, Q&A and chat

Most of this category should be explanatory answers from non-programming sites, not Stack Overflow code. Useful named communities include cooking, travel, history, philosophy, parenting, workplace, personal finance, writing and English usage. Technical answers supply a smaller complementary stratum. Take the answer as detector text and preserve its question separately for later topic matching.

IRC and wiki discussions add genuine turn-taking and informal disagreement. Preserve conversations coherently; concatenating unrelated short messages creates an artificial genre. This open-source mixture still lacks the breadth of everyday personal social-media posts. Describe that limitation rather than claiming exact social-domain equivalence. Use post/revision-level attribution and applicable historical license versions. [Stack Exchange licensing](https://stackoverflow.com/help/licensing), [Ubuntu IRC archive](https://irclogs.ubuntu.com/), [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

### General web and mixed

Define this as the **residual general-prose category** after routing identifiable news, papers, fiction, reviews and reference pages into their proper categories. A web archive is a retrieval source, not a writing genre. Without this routing, a large crawler dataset quietly duplicates the other eight pools.

Creative Commons Common Crawl is the most useful scalable discovery route here. The pinned metadata identifies 570 candidate files in 45 crawl partitions from 2013–2021. Those are files to inspect, not 570 cleared data shards. EFF, Foodista and Wikivoyage provide named anchors for opinion/explanation, everyday food writing and travel. Require a license that applies to the text, not merely a CC image embedded on the page. [CCCC collection instructions](https://github.com/r-three/common-pile/tree/main/sources/cccc), [EFF copyright](https://www.eff.org/copyright).

### News

Mix professional reporting, community reporting, international stories and a modest science-journalism slice. Global Voices helps regional breadth; VOA requires excluding wire-service copies; Wikinews adds a different editorial process. OANC's historically contributed Slate files add US magazine prose under the corpus's release conditions. That grant does not extend to scraping today's Slate site. [Global Voices terms](https://globalvoices.org/about/global-voices-attribution-policy/), [VOA terms](https://www.voanews.com/p/5338.html), [Wikinews copyright](https://en.wikinews.org/wiki/Wikinews:Copyright), [OANC](https://anc.org/data/oanc/).

Group syndicated or translated copies and articles covering the same event before splitting. Label reporting, commentary and criticism separately. Do not substitute news summaries from summarization benchmarks for original news articles. SciDev's policy pages use different CC BY version numbers, so the registry preserves the ambiguity and calls for article-level evidence. [SciDev policies](https://www.scidev.net/content/policies.html).

### Essays and academic writing

A useful pool needs school argumentation, English learners and university disciplinary assignments. ASAP 2.0, ELLIPSE, PERSUADE and BAWE cover different parts of that space. Selecting only high-scoring essays would discard exactly the human variation an authorship detector needs to tolerate. Keep the original writing, including errors; do not substitute machine corrections or teacher annotations.

ASAP 2.0 is an unusually promising CC BY candidate, but its substantial overlap with PERSUADE must be deduplicated. The inspected source does not establish exact writing dates, so release year and exam context alone are not a completed authorship audit. PERSUADE also has conflicting publisher license statements that need release-level reconciliation. BAWE supplies university writing under noncommercial access conditions; ELLIPSE's official license is NC-SA. [ASAP 2.0](https://github.com/scrosseye/ASAP_2.0), [PERSUADE repository](https://github.com/scrosseye/persuade_corpus_2.0), [PERSUADE publisher offering](https://the-learning-agency-lab.com/learning-exchange/persuade-dataset/), [BAWE](https://warwick.ac.uk/fac/soc/al/research/collections/bawe/), [ELLIPSE](https://github.com/scrosseye/ELLIPSE-Corpus).

### Professional and finance

Use original government reports, financial-policy explanations, development reports and contributed correspondence. This is more representative than filling the whole category with statutes, patents or tables. OANC ICIC is a useful small source for grant proposals and fundraising letters; scale it only as far as its independent documents permit. [OANC contents](https://anc.org/data/oanc/contents/).

GovReport supplies an established report collection; the Federal Reserve Board supplies explicitly identified government text; World Bank items require per-work license filtering. Corporate emails, management communications and private-sector financial narrative remain weakly covered. Public access to SEC filings or Enron emails is not by itself the provenance and rights evidence needed to close that gap. [GovReport](https://gov-report-data.github.io/), [Federal Reserve disclaimer](https://www.federalreserve.gov/disclaimer.htm), [World Bank repository terms](https://www.worldbank.org/ext/en/legal/terms-conditions/open-knowledge-repository).

## What to acquire first

1. **Structured, traceable sources:** historical PMC articles, eligible ACL/arXiv papers, Gutenberg work metadata, and GovReport originals. They give the cleanest initial tests of identity, dates and prose extraction.
2. **Versioned reference and discussion:** Wikimedia histories and historical Stack Exchange revisions. The main engineering task is obtaining the actual pre-cutoff text, not filtering today's text on creation date.
3. **Licensed web and news:** the listed historical CCCC partitions and named publishers. Audit a small sample from every proposed domain before bulk collection.
4. **Small high-value style sources:** SCP tales, OANC correspondence/journalism and eligible school essays. Check independent source counts before increasing their exposure.
5. **Conditional slots:** resolve review and WritingPrompts rights, learner-corpus restrictions, and the PERSUADE release discrepancy. Preserve their allocated space until then.

This order is about intake reliability, not changing the final nine-category proportions. A 100,000-passage design is a manageable accounting example, not a justified optimal training size. Estimate cost and storage from a measured pilot; web pages, compressed JSONL and PDFs have very different acquisition costs.

## Deliverables and reproducibility

- [Detailed source registry](SOURCE_REGISTRY.md): 48 source entries, including 11 zero-weight alternatives; access links, evidence, selection rules and remaining checks for each.
- [Structured registry](source-registry.json): the same source decisions in machine-readable form.
- [Sampling and admission specification](SAMPLING.md): document provenance, deduplication, quarantine, split protection and quality-control rules.
- [100,000-passage allocation](sampling-plan.json): integer category/source quotas with missing availability explicitly represented.
- [Candidate file inventory](candidate-shards.json): exact pinned historical CCCC file paths and selected Stack Exchange files. Stack Exchange still needs row/revision date checks.
- [Evidence manifest](evidence-manifest.json): 20 small downloaded metadata/documentation files with retrieval timestamps and hashes; 14 Hugging Face dataset revisions are pinned.
- [Validation record](validation.json): allocation totals and evidence hash checks.

Run `python3 research/human-source-pool/build_plan.py` from the project root to validate and regenerate the plan and readable registry. An optional `--size` changes the planning total; it never downloads text or repeats records to fill a quota. `collect_evidence.py` refreshes public metadata and should be treated as a new evidence snapshot, not silently mixed into a frozen collection.

Common Pile is useful collection infrastructure, but its ready-made pretraining mixture is not this detector mixture. Use selected sources and their original records; do not import its code, ASR transcripts or unverified modern text as ordinary human prose. [Common Pile paper](https://arxiv.org/html/2506.05209v1), [collection code](https://github.com/r-three/common-pile).

The project's existing paper-focused experiments remain separate. Earlier broad-domain trials did not establish gains for paper localization, so matching Pangram's source chart should be evaluated as a new data-design hypothesis, not treated as an automatic model improvement. See [research priorities](../TRAINING_RESEARCH_PRIORITIES.md).
