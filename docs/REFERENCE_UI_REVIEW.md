# Pangram results interaction review — 2026-09-23

Inspected the authenticated https://www.pangram.com/dashboard using its free mixed-text sample (credits remained at 20). No private documents were submitted.

Observed and implemented:
- Inline highlights respond with an underline shadow; measured transition is box-shadow 300 ms cubic-bezier(.4,0,.2,1). AI default fill rgba(255,86,48,.1); hover underline #ff4d0648; selected underline #ff4d0696.
- Clicking a highlighted segment selects Details and expands the associated segment card.
- Segment cards include category, word count, text preview and expanded result gauge.
- Details includes a document-wide classification chart and category filter.
- Hover information identifies the segment; our tooltip uses the available model score rather than inventing confidence.
- Selecting a segment card locates its text. Added keyboard activation and previous/next navigation.

Local adaptations:
- MELD has evidence scores and exploratory token/sentence localization, not Pangram’s calibrated human/AI-assisted/AI categories or confidence levels. These stay explicitly labelled as evidence.
- Scores are displayed out of 100, never as the proportion of AI-written words.
- Large reports use a sampled chart (labelled as such) and bounded segment-list pages. All saved segments remain accessible.
- Archive text stays read-only and opening these views never reruns classification.

Other observed gaps / limits:
- Pangram’s original chart uses human/assisted/AI categorical levels and its overview describes the distribution of AI regions. Matching its model-dependent conclusions requires compatible classifier output; they are not fabricated for MELD.
- Plagiarism is an upgrade-gated service on the reference account. The local application compares a reference corpus, not Pangram’s web plagiarism index.
- Pangram has email-when-ready, product integrations and subscription controls. These external services are not supplied by this visual update.

Verification: local cached MELD report, inline selection, matching expanded card, previous/next navigation, category filter, narrow-screen layout; TypeScript and production build checks.
