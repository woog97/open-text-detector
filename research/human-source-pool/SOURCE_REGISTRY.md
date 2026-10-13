# Human source registry

Researched October 1, 2026. All allocations below are proposals. No training passages have been admitted.

Read README.md for the category design and SAMPLING.md for admission and splitting rules. An open-subset designation is source-level evidence, not a completed document audit.

## Creative writing

### Project Gutenberg, with Standard Ebooks preferred for matching editions

**Allocation:** 45% within category; 10,000 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `gutenberg`.

Gutenberg has public-domain and some copyrighted works; Standard Ebooks supplies carefully edited public-domain editions. Rights are work- and jurisdiction-specific.

**Selection:** English narrative fiction; stratify genre, era, author and original versus translated English. Select the best verified transcription of each work, never count editions as independent. Prefer historical archived editions; PG-19 train can supply a reproducible fallback after checking its substitutions.

**Access:** [source](https://www.gutenberg.org/); [data or retrieval route](https://huggingface.co/datasets/common-pile/project_gutenberg).

**Evidence:** [source 1](https://www.gutenberg.org/policy/permission.html); [source 2](https://www.gutenberg.org/policy/robot_access.html); [source 3](https://standardebooks.org/).

**Group before splitting:** canonical_work_id, author_id, translation_id.

**Remaining checks:** Exclude the locally evaluated PG-19 test works and related editions. Check publication and transcription history; remove headers and editorial additions. Contemporary fiction remains undercovered.

**Related route:** [alternative 1](https://github.com/google-deepmind/pg19).

### WritingPrompts original 2018 release

**Allocation:** 30% within category; 6,667 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `writingprompts`.

The authors provide original prompt/story splits alongside separate model-generated samples. The README does not establish a commercial license for the underlying Reddit stories.

**Selection:** Original training stories only; separate prompt from story. Preserve paragraph structure when recoverable. Stratify genre and length, including ordinary prose rather than only highly voted stories.

**Access:** [source](https://github.com/facebookresearch/fairseq/tree/main/examples/stories); [data or retrieval route](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz).

**Evidence:** [source 1](https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/stories/README.md).

**Group before splitting:** prompt_thread_id, story_id, author_id.

**Remaining checks:** Hold this quota pending text rights. Reconstruct missing identity metadata where possible; otherwise quarantine uncertain groups. Remove all overlap with existing detector benchmarks. Never import generated samples.

### SCP Foundation prose tales

**Allocation:** 20% within category; 4,444 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `scp`.

SCP documents a CC BY-SA 3.0 framework with attribution and explicit exceptions for some non-text assets.

**Selection:** Exact pre-2022 revisions of English tales; favor narrative stories over repetitive containment templates. Cap this fictional universe at its allocated share.

**Access:** [source](https://scp-wiki.wikidot.com/); [data or retrieval route](https://scp-wiki.wikidot.com/tales-hub).

**Evidence:** [source 1](https://scp-wiki.wikidot.com/licensing-guide).

**Group before splitting:** page_id, series_or_canon, author_id.

**Remaining checks:** Verify historical revision retrieval and text license; omit images, third-party quotations, credits and navigation. Current page creation date alone is insufficient.

### Wikisource proofread short fiction

**Allocation:** 5% within category; 1,111 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wikisource`.

Wikisource hosts public-domain and freely licensed texts with work-specific copyright information.

**Selection:** Proofread or validated fiction and short stories with scans; prioritize works absent from Gutenberg. Exclude verse, drama and nonfiction from this first prose mixture.

**Access:** [source](https://en.wikisource.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://en.wikisource.org/wiki/Wikisource:Copyright_policy).

**Group before splitting:** canonical_work_id, author_id, translation_id.

**Remaining checks:** Preserve scan edition and proofreading evidence; deduplicate anthologies, excerpts and translations; distinguish historical text from recent annotations.

## Scientific and medical

### PMC Open Access commercial-reuse subset

**Allocation:** 35% within category; 6,306 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `pmc`.

PMC availability is not blanket permission; the OA subset exposes article-level reuse terms and supported machine-access routes.

**Selection:** CC BY or CC0 articles with verified pre-2022 text versions; extract body sections from JATS. Balance journals, article types and specialties; keep abstracts a minority.

**Access:** [source](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/); [data or retrieval route](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/).

**Evidence:** [source 1](https://pmc.ncbi.nlm.nih.gov/about/copyright/); [source 2](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/).

**Group before splitting:** doi, pmcid, canonical_paper_id, author_ids.

**Remaining checks:** Exclude corrected post-cutoff text, references, notices, quoted copyrighted material and benchmark ancestors. Deduplicate against arXiv, ACL and peS2o.

### arXiv full texts with suitable article licenses

**Allocation:** 30% within category; 5,405 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `arxiv`.

arXiv licenses vary by article; its metadata license is distinct from full-text rights.

**Selection:** CC BY, CC BY-SA or CC0 full texts; exact versions submitted by 2021-12-31. Balance mathematics, physics, CS, engineering, economics and statistics; do not build an abstracts-only pool.

**Access:** [source](https://info.arxiv.org/help/license/index.html); [data or retrieval route](https://huggingface.co/datasets/common-pile/arxiv_papers).

**Evidence:** [source 1](https://info.arxiv.org/help/license/index.html).

**Group before splitting:** arxiv_base_id, doi, canonical_paper_id, author_ids.

**Remaining checks:** Default arXiv distribution permission is not a blanket commercial adaptation license. Parse TeX without executing it. Verify version-level dates and paper-family overlaps.

### Common Pile licensed peS2o subset

**Allocation:** 20% within category; 3,604 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `pes2o`.

The Common Pile peS2o release is a separately curated source with its own metadata; it must not be conflated with every peS2o or S2ORC release.

**Selection:** Use for fields missing from PMC/arXiv/ACL: environmental science, engineering, psychology, education and social sciences. Require original document license evidence and a pre-2022 text version.

**Access:** [source](https://github.com/r-three/common-pile/tree/main/sources/pes2o); [data or retrieval route](https://huggingface.co/datasets/common-pile/peS2o).

**Evidence:** [source 1](https://huggingface.co/datasets/common-pile/peS2o/blob/main/README.md).

**Group before splitting:** corpus_id, doi, canonical_paper_id, author_ids.

**Remaining checks:** Do not rely on an open-access flag alone; audit PDF extraction, page order and source dates. Unknown version date remains quarantined.

### ACL-owned proceedings from 2016–2021

**Allocation:** 15% within category; 2,703 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `acl`.

ACL states CC BY 4.0 for its material from 2016 onward and CC BY-NC-SA 3.0 for earlier material; third-party proceedings need separate checks.

**Selection:** Human prose from 2016–2021 papers across NLP subfields and paper sections, excluding all existing local paper families. Retain citation metadata separately.

**Access:** [source](https://aclanthology.org/); [data or retrieval route](https://github.com/acl-org/acl-anthology).

**Evidence:** [source 1](https://aclanthology.org/faq/copyright/).

**Group before splitting:** anthology_id, doi, canonical_paper_id, author_ids.

**Remaining checks:** Apply per-paper ownership checks. Existing local 2013–2021 paper data is not automatically cleared under the newer license.

## Reference and educational

### English Wikipedia historical article revisions

**Allocation:** 50% within category; 7,958 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wikipedia`.

Wikimedia provides licensed text and database dumps. Revision history is needed to recover historical text rather than merely historical page creation dates.

**Selection:** Latest eligible revision at or before 2021-12-31. Broad subject strata; take lead and body prose. Exclude lists, disambiguation, templates, copyvio pages and bot-created stubs.

**Access:** [source](https://en.wikipedia.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

**Group before splitting:** page_id, wikidata_entity_id, revision_family.

**Remaining checks:** Retain attribution/history and applicable license version. If a mirror lacks revision timestamps, use a dated dump or history reconstruction instead.

### Pressbooks openly licensed textbook editions

**Allocation:** 25% within category; 3,979 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `pressbooks`.

Pressbooks Directory exposes license filters and includes both open and restricted books.

**Selection:** CC BY, CC BY-SA or CC0 editions from before 2022. Balance humanities, social sciences, health, STEM and vocational education; choose explanatory prose with clean HTML or EPUB.

**Access:** [source](https://pressbooks.directory/); [data or retrieval route](https://huggingface.co/datasets/common-pile/pressbooks).

**Evidence:** [source 1](https://pressbooks.directory/); [source 2](https://pressbooks.com/open-education/introducing-pressbooks-directory/).

**Group before splitting:** canonical_book_id, edition_family, chapter_id, author_ids.

**Remaining checks:** A catalog hit is not a text grant; inspect book and chapter notices. Merge clones, remixes and duplicate OER catalog entries. Historical edition evidence required.

### LibreTexts selected book and page licenses

**Allocation:** 15% within category; 2,387 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `libretexts`.

The Common Pile provides a LibreTexts source. License and source provenance still need checking on the particular book or page.

**Selection:** Historical CC BY/CC BY-SA explanatory prose, with material beyond chemistry and biology. Keep derivations only when accompanied by readable prose.

**Access:** [source](https://libretexts.org/); [data or retrieval route](https://huggingface.co/datasets/common-pile/libretexts).

**Evidence:** [source 1](https://huggingface.co/datasets/common-pile/libretexts/blob/main/README.md).

**Group before splitting:** canonical_book_id, page_id, remix_family.

**Remaining checks:** Many reused textbooks have different licenses. Exclude NC/ND for the default pool; remove exercise keys and table-only content.

### English Wikibooks historical chapters

**Allocation:** 10% within category; 1,592 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wikibooks`.

Wikibooks text is part of the Wikimedia licensing and attribution framework.

**Selection:** Pre-2022 revisions of coherent explanatory chapters. Favor practical and humanities material underrepresented in the other reference sources.

**Access:** [source](https://en.wikibooks.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

**Group before splitting:** book_id, chapter_family, revision_family.

**Remaining checks:** Exclude incomplete stubs, copied Wikipedia pages and code-dominated chapters. Keep whole-book families in one split.

### OpenStax historical editions

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** version_specific. **ID:** `openstax`.

Current official licensing guidance lists CC BY-NC-SA. Older CC BY editions need their own exact content and license evidence.

**Selection:** Recover a verifiable old edition before admitting it; do not label the entire present catalog CC BY.

**Access:** [source](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks); [data or retrieval route](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks).

**Evidence:** [source 1](https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### OpenIntro Statistics

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** version_specific. **ID:** `openintro`.

A concrete educational text candidate; license must be checked inside the selected historical edition.

**Selection:** Useful editorial-quality anchor, too few independent titles to dominate a large pool.

**Access:** [source](https://www.openintro.org/book/os/); [data or retrieval route](https://www.openintro.org/book/os/).

**Evidence:** [source 1](https://www.openintro.org/book/os/).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

## Consumer reviews

### Amazon Review Data 2018

**Allocation:** 50% within category; 5,306 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `amazon2018`.

The original release reports 233.1 million reviews from May 1996–October 2018 with review, product and user metadata. Public download alone does not establish rights to the review text.

**Selection:** Sample original review text from full per-category files, not ratings-only or five-core-only subsets. Cover home, apparel, electronics, books, beauty, outdoor, toys and other products; include middle ratings.

**Access:** [source](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html); [data or retrieval route](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html).

**Evidence:** [source 1](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html).

**Group before splitting:** reviewer_id, product_parent_id, review_id, duplicate_cluster.

**Remaining checks:** Hold commercial-use quota until underlying rights are documented. Fake, incentivized and templated reviews remain possible even before LLMs. Remove product descriptions and seller replies.

### Yelp Open Dataset historical reviews

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial. **ID:** `yelp`.

Yelp publishes an academic-use agreement. The dataset is not an unrestricted commercial-text release.

**Selection:** If appropriate rights are available, select archived pre-2022 reviews across restaurants, retail, hospitality and local services; stratify ratings, region and length.

**Access:** [source](https://www.yelp.com/dataset); [data or retrieval route](https://www.yelp.com/dataset).

**Evidence:** [source 1](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf).

**Group before splitting:** user_id, business_id, review_id, duplicate_cluster.

**Remaining checks:** Retain a historical snapshot, not just an old date on a current edited record. Academic eligibility and permitted uses must match the actual project.

### Stanford Large Movie Review Dataset

**Allocation:** 10% within category; 1,061 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `imdb`.

The 2011 corpus contains labeled positive/negative reviews and an additional unlabeled split. The labeled selection is polarized.

**Selection:** Use eligible training and unlabeled originals after benchmark exclusion, with a small quota because long film criticism differs from typical product reviews.

**Access:** [source](https://nlp.stanford.edu/~amaas/data/sentiment/index.html); [data or retrieval route](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

**Evidence:** [source 1](https://nlp.stanford.edu/~amaas/data/sentiment/index.html).

**Group before splitting:** imdb_review_url, movie_id, reviewer_id.

**Remaining checks:** Verify underlying text rights; do not import the official test split or benchmark copies. Positive/negative balance is not the natural consumer rating distribution.

### OpinRank hotel and car reviews

**Allocation:** 40% within category; 4,244 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `opinrank`.

UCI labels this 2011 collection CC BY 4.0; its text originates from Tripadvisor and Edmunds. The upstream grant still needs verification.

**Selection:** Hotel and automotive narratives diversify the review category. Balance cities and product types and retain original dates and review IDs.

**Access:** [source](https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset); [data or retrieval route](https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset).

**Evidence:** [source 1](https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset).

**Group before splitting:** review_id, author_id, hotel_or_car_id.

**Remaining checks:** Do not infer that a repository license settles the rights of upstream reviewers. Audit extraction and duplicate review fields.

### Amazon multilingual review corpus

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial. **ID:** `amazon_marc`.

The official license limits use to academic research and forbids commercial use.

**Selection:** Potential multilingual research comparator, not a default replacement for missing commercial review rights.

**Access:** [source](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt); [data or retrieval route](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt).

**Evidence:** [source 1](https://github.com/awslabs/open-data-docs/blob/main/docs/amazon-reviews-ml/license.txt).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### Amazon Reviews 2023

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `amazon2023`.

The release extends through September 2023; old timestamps do not prove unchanged historical text.

**Selection:** Prefer the 2018 snapshot for human provenance; investigate this only for coverage gaps with version checks.

**Access:** [source](https://amazon-reviews-2023.github.io/); [data or retrieval route](https://amazon-reviews-2023.github.io/).

**Evidence:** [source 1](https://amazon-reviews-2023.github.io/).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

## Social, Q&A, chat

### Stack Exchange non-programming answers

**Allocation:** 55% within category; 5,726 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `stack_nontech`.

Public contributions use CC BY-SA with license versions changing over time. Dump access arrangements have also changed.

**Selection:** Pre-2022 answer revisions from cooking, travel, history, philosophy, parenting, workplace, personal finance, writing and English usage. Use whole answers; store questions as context rather than prepend to detector text.

**Access:** [source](https://stackoverflow.com/help/licensing); [data or retrieval route](https://huggingface.co/datasets/common-pile/stackexchange).

**Evidence:** [source 1](https://stackoverflow.com/help/licensing); [source 2](https://meta.stackexchange.com/questions/401324/announcing-a-change-to-the-data-dump-process).

**Group before splitting:** site, thread_id, answer_id, author_id.

**Remaining checks:** Preserve post-level attribution, dates and license. Remove quotations and code without rewriting prose. Current dumps can contain post-cutoff edits.

### Stack Exchange technical explanations

**Allocation:** 20% within category; 2,082 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `stack_tech`.

The same contribution-license framework covers technical Stack Exchange sites.

**Selection:** Explanatory prose from science, mathematics, statistics and computing sites. Cap code and formula dominance; retain accessible explanations and uncertain or imperfect answers.

**Access:** [source](https://stackoverflow.com/help/licensing); [data or retrieval route](https://huggingface.co/datasets/common-pile/stackexchange).

**Evidence:** [source 1](https://stackoverflow.com/help/licensing).

**Group before splitting:** site, thread_id, answer_id, author_id.

**Remaining checks:** Use disjoint site/post routing from the nontechnical quota. Do not select solely by votes or detector score.

### Ubuntu IRC historical conversations

**Allocation:** 15% within category; 1,561 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `ubuntu_irc`.

The Common Pile publishes a source assembled from historical Ubuntu IRC logs.

**Selection:** Pre-2022 coherent conversations with intact turn boundaries; exclude bot output, commands, server events and unrelated concatenated messages. Retain a distinct chat stratum.

**Access:** [source](https://irclogs.ubuntu.com/); [data or retrieval route](https://huggingface.co/datasets/common-pile/ubuntu_irc).

**Evidence:** [source 1](https://huggingface.co/datasets/common-pile/ubuntu_irc/blob/main/README.md).

**Group before splitting:** channel, session_id, participant_ids.

**Remaining checks:** Verify public-domain release basis in source documentation and event dates. Thread segmentation is harder than for Q&A; exclude uncertain context. Mostly technical community, not broad social media.

### Wikipedia human discussion threads

**Allocation:** 10% within category; 1,041 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wiki_talk`.

Article talk pages provide discussion rather than encyclopedia prose under Wikimedia text terms.

**Selection:** Pre-2022 signed discussion turns grouped into threads; retain disagreements and informal reasoning. Remove bot notices, signatures from model text, and extensive quotations.

**Access:** [source](https://en.wikipedia.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

**Group before splitting:** page_id, thread_id, participant_ids.

**Remaining checks:** History and thread boundaries required; related article topic families must not cross protected splits. Do not count these as encyclopedic reference text.

## General web and mixed

### Common Pile Creative Commons Common Crawl

**Allocation:** 65% within category; 5,335 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `cccc`.

The source is organized by Common Crawl snapshot; its collection pipeline includes license tagging and domain review.

**Selection:** Only crawl snapshots through 2021, with document-level license evidence. Route scientific, reference, reviews and news out before filling this residual category; retain blogs, hobbies, how-to and personal nonfiction.

**Access:** [source](https://github.com/r-three/common-pile/tree/main/sources/cccc); [data or retrieval route](https://huggingface.co/datasets/common-pile/cccc).

**Evidence:** [source 1](https://huggingface.co/datasets/common-pile/cccc/blob/main/README.md); [source 2](https://raw.githubusercontent.com/r-three/common-pile/main/sources/cccc/README.md).

**Group before splitting:** canonical_url, registered_domain, author_id, duplicate_cluster.

**Remaining checks:** License markers can describe an image rather than text. Audit domains and extraction; cap any one domain at 5% of this category. Do not use a recent crawl merely because the page claims an old date.

### EFF historical Deeplinks articles

**Allocation:** 10% within category; 821 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `eff`.

EFF currently licenses original material CC BY 4.0 unless otherwise noted and distinguishes third-party content.

**Selection:** Archive-proven pre-2022 original explainers and opinion pieces. Limit this technology-policy voice to the assigned small share.

**Access:** [source](https://www.eff.org/deeplinks); [data or retrieval route](https://www.eff.org/deeplinks).

**Evidence:** [source 1](https://www.eff.org/copyright).

**Group before splitting:** canonical_url, author_id, series_id.

**Remaining checks:** Check historical content and license notice, quotations and republications. Retain clear bylines; do not include action widgets or footers.

### Foodista narrative food writing

**Allocation:** 10% within category; 821 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `foodista`.

A dedicated Common Pile Foodista source provides a concrete route to food-related web text.

**Selection:** Pre-2022 narrative introductions, original food articles and explanatory cooking prose. Retain title/recipe context separately; exclude ingredient-only lists.

**Access:** [source](https://www.foodista.com/); [data or retrieval route](https://huggingface.co/datasets/common-pile/foodista).

**Evidence:** [source 1](https://huggingface.co/datasets/common-pile/foodista/blob/main/README.md).

**Group before splitting:** canonical_url, author_id, recipe_family.

**Remaining checks:** Check source-level and document-level license notices and historical version dates. Keep user comments separate; do not pad prose with ingredients to reach a word threshold.

### Wikivoyage travel prose

**Allocation:** 15% within category; 1,231 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wikivoyage`.

Wikivoyage is a Wikimedia travel-writing source with reusable text under its applicable terms.

**Selection:** Pre-2022 destination and practical-travel prose, broad region balance, without listing tables or coordinates. Assign here only once.

**Access:** [source](https://en.wikivoyage.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).

**Group before splitting:** page_id, destination_family, revision_family.

**Remaining checks:** Descriptions are not consumer reviews. Deduplicate Wikitravel imports and related mirrors; verify revision-level attribution.

### FineWeb / FineWeb-Edu historical snapshots

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** rights_unresolved. **ID:** `fineweb_edu`.

ODC-By is a compilation license, not a blanket grant for webpage text. Educational filtering is not human-authorship certification.

**Selection:** Use only if upstream rights and pre-2022 capture evidence can be established; broad web is preferable to only high educational scores.

**Access:** [source](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); [data or retrieval route](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

**Evidence:** [source 1](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### MASC manually annotated corpus

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `masc`.

The project states CC BY 3.0 US and offers about 500,000 words across genres.

**Selection:** Best as a small extraction-quality audit; overlaps OANC, so never count it as independent scale.

**Access:** [source](https://anc.org/data/masc/); [data or retrieval route](https://anc.org/data/masc/).

**Evidence:** [source 1](https://anc.org/data/masc/).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### Common Pile source collection

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** collection_only. **ID:** `common_pile`.

A source collection and collection-code route, not an extra independent donor.

**Selection:** Use selected sources only; exclude code, patents, ASR transcripts and modern unverified text from this proposed written-prose pool.

**Access:** [source](https://huggingface.co/collections/common-pile/common-pile-v01-raw-data); [data or retrieval route](https://huggingface.co/collections/common-pile/common-pile-v01-raw-data).

**Evidence:** [source 1](https://huggingface.co/collections/common-pile/common-pile-v01-raw-data).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

## News

### Global Voices original English news

**Allocation:** 25% within category; 1,652 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `globalvoices`.

Global Voices specifies CC BY 3.0 for its original content; third-party content is treated separately.

**Selection:** Pre-2022 original English reporting and identified human translations, with wide regional coverage. Keep translations linked to the same story family.

**Access:** [source](https://globalvoices.org/); [data or retrieval route](https://globalvoices.org/).

**Evidence:** [source 1](https://globalvoices.org/about/global-voices-attribution-policy/).

**Group before splitting:** canonical_story_id, translation_family, author_id, event_cluster.

**Remaining checks:** Quoted social posts are not automatically covered by the outlet license. Preserve publication and capture evidence; remove embedded posts when rights/provenance are uncertain.

### Voice of America original reporting

**Allocation:** 25% within category; 1,652 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `voa`.

VOA distinguishes its own public-domain productions from third-party wire content.

**Selection:** Historical original VOA-authored text only; balance geography and subject. Require byline/source evidence to distinguish Reuters, AP and AFP copies.

**Access:** [source](https://www.voanews.com/); [data or retrieval route](https://www.voanews.com/).

**Evidence:** [source 1](https://www.voanews.com/p/5338.html).

**Group before splitting:** canonical_article_id, author_id, event_cluster, syndication_cluster.

**Remaining checks:** Reject mixed or unclear wire attribution; dates need historical version/capture support. Outlet policy is not permission for every hosted article.

### Wikinews archived reporting

**Allocation:** 15% within category; 991 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `wikinews`.

Wikinews uses different text-license regimes across dates; historical articles are not uniformly covered by the current CC BY version.

**Selection:** Pre-2022 published article revisions; prioritize reviewed reporting over incomplete drafts and short briefs. Keep article families and quoted source material identifiable.

**Access:** [source](https://en.wikinews.org/); [data or retrieval route](https://dumps.wikimedia.org/backup-index.html).

**Evidence:** [source 1](https://en.wikinews.org/wiki/Wikinews:Copyright).

**Group before splitting:** article_id, revision_family, event_cluster.

**Remaining checks:** Record the applicable historical license, usually CC BY 2.5 for this period; exclude third-party quotations where needed.

### SciDev.Net original science and development journalism

**Allocation:** 15% within category; 991 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `scidev`.

SciDev.Net publishes CC BY reuse policies, with different version numbers on its policy and media pages.

**Selection:** Pre-2022 reporting written for general readers; balance regions and avoid filling the scientific-paper category with these stories.

**Access:** [source](https://www.scidev.net/); [data or retrieval route](https://www.scidev.net/).

**Evidence:** [source 1](https://www.scidev.net/content/policies.html); [source 2](https://www.scidev.net/content/media.html).

**Group before splitting:** canonical_article_id, translation_family, author_id, event_cluster.

**Remaining checks:** Pin article-level license and historical text; distinguish guest/third-party material. Do not silently resolve conflicting policy-version numbers.

### OANC contributed Slate journalism

**Allocation:** 20% within category; 1,321 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `oanc_slate`.

OANC states unrestricted commercial use for its released subset and includes historical Slate material. This does not grant rights to arbitrary Slate articles.

**Selection:** Use only the contributed OANC files; stratify topics and separate opinion/criticism from reporting. Preserve original article metadata.

**Access:** [source](https://anc.org/data/oanc/); [data or retrieval route](https://anc.org/data/oanc/download/).

**Evidence:** [source 1](https://anc.org/data/oanc/); [source 2](https://anc.org/data/oanc/contents/).

**Group before splitting:** oanc_doc_id, canonical_url, author_id.

**Remaining checks:** Audit the release notices; deduplicate MASC and other OANC-derived copies. Historical US magazine prose is a limited slice, not current global news.

### European Commission Horizon magazine

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `horizon`.

The publisher states CC BY 4.0 for articles.

**Selection:** Historical science journalism backup for SciDev quota after edition checks; not general-news replacement.

**Access:** [source](https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/republish-our-stories); [data or retrieval route](https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/republish-our-stories).

**Evidence:** [source 1](https://projects.research-and-innovation.ec.europa.eu/en/horizon-magazine/republish-our-stories).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

## Essays and academic writing

### ASAP 2.0 original training essays

**Allocation:** 60% within category; 2,943 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `asap2`.

The publisher states CC BY 4.0 and 24,278 source-based student essays over seven prompts. There is substantial PERSUADE overlap; its reported component counts do not sum to the headline count.

**Selection:** Train split only, original essay text and all score bands. Balance prompts, grades and documented ELL status without feeding demographic fields to the detector.

**Access:** [source](https://github.com/scrosseye/ASAP_2.0); [data or retrieval route](https://github.com/scrosseye/ASAP_2.0/blob/main/ASAP_2_Final_github_train.zip).

**Evidence:** [source 1](https://raw.githubusercontent.com/scrosseye/ASAP_2.0/main/README.md); [source 2](https://zenodo.org/records/14781349).

**Group before splitting:** essay_id, student_id, prompt_source_family, duplicate_cluster.

**Remaining checks:** Exact collection dates and permitted writing tools were not established by the inspected preprint. Resolve human provenance before admission. Match overlapping PERSUADE essays and exclude every benchmark family.

### ELLIPSE original reliable training essays

**Allocation:** 25% within category; 1,226 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial. **ID:** `ellipse`.

The official release describes about 6,500 reliably scored learner essays and a CC BY-NC-SA 4.0 license. A larger raw collection includes less reliable items.

**Selection:** If permitted, use original train essays across proficiency bands, without proofreading or language-model cleanup; retain one row per essay.

**Access:** [source](https://github.com/scrosseye/ELLIPSE-Corpus); [data or retrieval route](https://github.com/scrosseye/ELLIPSE-Corpus).

**Evidence:** [source 1](https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/README.md).

**Group before splitting:** essay_id, student_id, prompt_source_family.

**Remaining checks:** Local ELLIPSE evaluation already exists. Exclude its protected originals and related prompts; do not automatically promote local cached data into training. Check collection dates.

### British Academic Written English

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial. **ID:** `bawe`.

BAWE contains 2,761 assessed university texts from 2004–2007 across disciplines; its published access description is for noncommercial researchers.

**Selection:** If permitted, sample humanities, social sciences, life sciences and physical sciences; include undergraduate and masters work. Remove assignment prompts and instructor annotations.

**Access:** [source](https://warwick.ac.uk/fac/soc/al/research/collections/bawe/); [data or retrieval route](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2539).

**Evidence:** [source 1](https://warwick.ac.uk/fac/soc/al/research/collections/bawe/).

**Group before splitting:** student_id, assignment_id, document_family.

**Remaining checks:** Registration and actual use terms need satisfaction. These are proficient assessed texts, so they cannot replace learner essays. Check references and quoted assigned readings.

### PERSUADE original essays with version-specific rights

**Allocation:** 15% within category; 736 passages in the 100,000-passage planning example. **Rights evidence:** conflicting. **ID:** `persuade`.

The PERSUADE 2.0 repository says CC BY-NC-SA 4.0, while The Learning Agency Lab labels its PERSUADE offering CC BY 4.0. These may describe different releases.

**Selection:** Prioritize independent-prompt essays absent from ASAP 2.0 to add rhetorical variety. Resolve exact file/release and license before choosing rows.

**Access:** [source](https://github.com/scrosseye/persuade_corpus_2.0); [data or retrieval route](https://the-learning-agency-lab.com/learning-exchange/persuade-dataset/).

**Evidence:** [source 1](https://raw.githubusercontent.com/scrosseye/persuade_corpus_2.0/main/README.md); [source 2](https://the-learning-agency-lab.com/learning-exchange/persuade-dataset/).

**Group before splitting:** essay_id, student_id, prompt_source_family, duplicate_cluster.

**Remaining checks:** Do not transfer the more permissive statement to all versions. Exclude duplicate annotation rows and ASAP overlap; establish collection-date provenance.

### PELIC learner corpus

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial_no_derivatives. **ID:** `pelic`.

The release uses CC BY-NC-ND 4.0. Local PELIC evaluation data also exists.

**Selection:** Restricted alternative for longitudinal learner writing; verify use permission, student grouping and benchmark separation.

**Access:** [source](https://github.com/ELI-Data-Mining-Group/PELIC-dataset); [data or retrieval route](https://github.com/ELI-Data-Mining-Group/PELIC-dataset).

**Evidence:** [source 1](https://github.com/ELI-Data-Mining-Group/PELIC-dataset).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### BEA 2019 Write & Improve / LOCNESS

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** noncommercial. **ID:** `write_improve`.

The task page specifies noncommercial use for these learner corpora.

**Selection:** Retain original learner sentences, not corrected or system outputs; reconstruct document and student groups.

**Access:** [source](https://www.cl.cam.ac.uk/research/nl/bea2019st/); [data or retrieval route](https://www.cl.cam.ac.uk/research/nl/bea2019st/).

**Evidence:** [source 1](https://www.cl.cam.ac.uk/research/nl/bea2019st/).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

### Public Domain Review essays

**Allocation:** 0% within category; 0 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `public_domain_review`.

Most essays carry CC BY-SA, but some have more restrictive terms.

**Selection:** Small professionally edited essay anchor; distinguish it from student writing and identify quoted book extracts.

**Access:** [source](https://publicdomainreview.org/reusing-material); [data or retrieval route](https://publicdomainreview.org/reusing-material).

**Evidence:** [source 1](https://publicdomainreview.org/reusing-material).

**Group before splitting:** canonical_source_id, duplicate_cluster.

**Remaining checks:** Zero-weight alternative; no automatic substitution or admission.

## Professional and finance

### GovReport original CRS and GAO reports

**Allocation:** 35% within category; 1,086 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `govreport`.

GovReport offers a concrete 2021 release of long government reports paired with summaries.

**Selection:** Use original report bodies from eligible training files, with summary text stored separately. Balance policy topics and report sections, excluding tables and repeated agency boilerplate.

**Access:** [source](https://gov-report-data.github.io/); [data or retrieval route](https://gov-report-data.github.io/).

**Evidence:** [source 1](https://gov-report-data.github.io/).

**Group before splitting:** agency_report_id, update_series, report_family.

**Remaining checks:** Verify source-level government authorship and third-party inclusions. Exclude official benchmark test documents and all related update versions.

### Federal Reserve Board reports and policy prose

**Allocation:** 25% within category; 776 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `fed`.

The Board describes its site information as public domain unless indicated otherwise; third-party material is excluded.

**Selection:** Pre-2022 original monetary-policy, financial-stability and explanatory report prose. Use archived PDFs/HTML with identifiable dates and versions.

**Access:** [source](https://www.federalreserve.gov/); [data or retrieval route](https://www.federalreserve.gov/publications.htm).

**Evidence:** [source 1](https://www.federalreserve.gov/disclaimer.htm).

**Group before splitting:** report_series, edition_id, author_id.

**Remaining checks:** Do not extend the Board policy automatically to regional banks or third-party working papers. Remove tables and repeated release templates.

### World Bank openly licensed reports

**Allocation:** 25% within category; 776 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `worldbank`.

The repository contains multiple licenses, including CC BY and restricted NC/ND variants, with separate third-party content conditions.

**Selection:** Select historical CC BY or CC BY IGO reports with country and sector diversity. Prefer narrative findings, executive explanations and professional recommendations.

**Access:** [source](https://openknowledge.worldbank.org/); [data or retrieval route](https://openknowledge.worldbank.org/).

**Evidence:** [source 1](https://www.worldbank.org/ext/en/legal/terms-conditions/open-knowledge-repository).

**Group before splitting:** handle_id, report_series, translation_family, author_ids.

**Remaining checks:** Check each work and embedded material; do not substitute general website terms for the work license. Retain exact edition and archive evidence.

### OANC ICIC professional correspondence

**Allocation:** 15% within category; 465 passages in the 100,000-passage planning example. **Rights evidence:** open_subset. **ID:** `oanc_icic`.

The contributed ICIC portion includes fundraising letters, grant proposals, case statements and annual reports.

**Selection:** Use only the released OANC ICIC files. Retain natural persuasive and administrative prose, with names/contact details withheld from model text where appropriate.

**Access:** [source](https://anc.org/data/oanc/contents/); [data or retrieval route](https://anc.org/data/oanc/download/).

**Evidence:** [source 1](https://anc.org/data/oanc/); [source 2](https://anc.org/data/oanc/contents/).

**Group before splitting:** oanc_doc_id, organization_id, campaign_family.

**Remaining checks:** Small supply; do not repeat letters to fill quotas. Audit inherited release notices and overlap with MASC. Ordinary corporate email remains a coverage gap.
