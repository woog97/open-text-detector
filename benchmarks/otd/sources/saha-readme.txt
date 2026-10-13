## Policies Permitting LLM Use for Polishing Peer Reviews Are Currently Not Enforceable

This repository contains code and artifacts for our paper on detection of AI-generated peer reviews.

### Data

We collected peer reviews of conferences from pre-ChatGPT era, specifically from CoNLL 2016, ACL 2017, ICLR 2017, and NeurIPS 2013–2017, sourced from the PeerRead dataset. These papers are reviews are stored in `data/rawdata`. Starting from these reviews, we generate reviews with popular LLMs to simulate 4 levels of human-involvement. Following is the mapping of these levels (as they are called in the code) to how we describe them in the paper:

| Level | Description | Input to LLM |
|---|---|---|
| **Level 1** | AI-BP: AI-generated with Basic Prompts  | Paper + Reviewing guidelines
| **Level 2** | AI-EP: AI-generated with Elaborate Prompts | Paper + Reviewing guidelines + Conference-issued best practice documents
| **Level 3** | AI-HI: AI-generated with Human Input | Paper + Reviewing guidelines + Key assessment points as bulleted list
| **Level 4** | H-AI: Human written AI polished | Human written review 
| **reviews** | H: Fully Human written review | NA

There are two subsets, reviews in what we call **base subset** (more papers, fewer models, prompts) in the paper are present as individual files under `data/cleandata/`. Reviews under the **hard subset** (less papers, more models and prompts) are placed similarly under `data/cleandata/hard-subset`. Therefore, reviews in `data/cleandata` outside ``data/cleandata/hard-subset`` are from the easy subset. Additionally, humanized reviews generated with Undetectable AI are placed in `data/humanization/humanized_data`

### Pathfiles
These are text files where each line is a path to a review. Multiple scripts in this repository receive such a pathfile which specifies what set of reviews it will be operating on. `PathFiles` folder contains three such example pathfiles. `all_paths_easy.txt` corresponds to the easy subset, `all_paths_hard.txt` corresponds to the hard subset, `all_paths_humans.txt` to the set of human reviews and `all_paths_humanized.txt` to the humanized reviews.

### Generating LLM-assisted reviews

The conference reviewing guidelines are listed in `data-and-reference-generation/conference-guidelines.yaml`. Prompts for easy and hard subsets are listed in `data/data-and-reference-generation/base-subset-prompts.yaml` and `data/data-and-reference-generation/hard-subset-prompts.yaml` respectively. 

To generate reviews with open-source LLMs (appropriately hosted through a vllm server), run

```bash
python generate_new_data.py \
	--pathfile [PATHFILE] \
	--output_dir [DISTINATION_DIR] \
	--model_name [MODEL_PATH] \
	--base_url [IP_ADDR, PORT] \
	--global_openai_metadata [METADATA JSONL FILE] \
	--prompt_templates_file [YAML FILE WITH PROMPTS]
```

`PATHFILE` should contain path to all human reviews, whose LLM-assisted mirrors are to be generated. `[METADATA JSONL FILE]` can be any jsonl filepath where some auxiliary metadata related to the generation will be dumped. The generated reviews are saved at appropriate locations under `[DESTINATION_DIR]`.

The same script supports generating reviews with OpenAI and Gemini models with their batch APIs. For that, set the `OPENAI_API_KEY` (`GEMINI_API_KEY`) environment variable and run

```bash
python generate_new_data.py \
	--pathfile [PATHFILE] \
	--output_dir [DESTINATION DIR] \
	--model_name [MODEL ID] \
	--create_batch_file \
	--batch_file_path [BATCHFILE PATH] \
	--prompt_templates_file [YAML FILE WITH PROMPTS] \
	--base_url https://generativelanguage.googleapis.com/v1beta/openai/ # only for Gemini 
```

This creates the batch file at `BATCHFILE PATH` which is used in subsequent steps to submit the batch and retrieve the responses. For that, run

```bash
python submit_openai_batch_job.py \
	--batch_file_path [BATCHFILE PATH] \
	--description [DESCRIPTION] \
	--base_url https://generativelanguage.googleapis.com/v1beta/openai/ # only for Gemini 
```

With the alphanumeric `BATCH_ID` returned from the above step, run

```bash
python monitor_and_save_responses_openai_batch_job.py \
	--batch_id [BATCH_ID] \
	--batch_file_path [BATCHFILE PATH] \
	--save_output_path [BATCH RESPONSES DESTINATION] \
	--global_openai_metadata [METADATA JSONL FILE] \
	--base_url https://generativelanguage.googleapis.com/v1beta/openai/ # only for Gemini 
```

followed by

```bash
python save_files_from_batch_responses.py \
  --batch_responses [BATCH RESPONSES DESTINATION]
```

which saves the reviews individually at the appropriate locations.

Note that AI-HI (level3) generation prompts contains key assessment points from human written reviews. These key points are generated with GPT-5-mini and saved with the suffix `_keypoints` in the same locations as the human reviews. For generating the key points, we follow a similar batch pipeline described above (refer to `data/key-points-generation/`).

### Pangram & GPTZero

Set `PANGRAM_API_KEY` or `GPTZERO_API_KEY` depending on which detector you want to run, then substitute `[PATHFILE]` and `[DESTINATION_JSON_FILE]` with appropriate paths and run the corresponding scripts:

| Detector | Directory | Run | Analyse |
|----------|-----------|-----|---------|
| Pangram | `DetectorEval/Pangram/` |  `run_Pangram.py` | `analyse_pangram_hard_split.py` |
| GPTZero | `DetectorEval/GPTZero/` | `run_GPTZero.py` | `analyse_gptzero_hard_split.py` |
```bash
python <run_script> && python <analyse_script>
```

### Log Likelihood, Binoculars and FastDetectGPT

Set `HF_TOKEN`, then substitute `[PATHFILE]` and `[DESTINATION_JSON_FILE]` with appropriate paths and run the corresponding scripts:

| Detector | Run Script | Optional Args |
|----------|-----|---------|---------|
| Binoculars | `DetectorEval/Binoculars/run_binoculars.py` | `--context` |
| FastDetectGPT | `DetectorEval/FastDetectGPT/run_fastdetepy` |  `--context` |
| LogLikelihood | `DetectorEval/FastDetectGPT/run_fastdetect.py --sampling_model_name llama3-8b --scoring_model_name llama3-8b-instruct --only_log_likelihood` | `--context` |
```bash
python <run_script> <args>
```

### Similarity to AI-generated references

In this approach, we first generate several AI-generated reviews by prompting LLMs with the paper manuscript. We hypothesize that higher similarity with these references is an indicator of greater AI involvement.

To generate AI-generated reviews for each paper, follow the batch pipeline described in the data generation section (use `data/data-and-reference-generation/generate_references.py` for creating the batch file).

We use three measures of similarity, namely cosine similarity, soft-n-gram matching and soft-keypoint matching. To save similarity scores of reviews with their references, refer to the following files in `peer-review-advantages/Anchor`:

| Similarity metric | Run | Analyse |
|-------------------|-----|------------|
| Cosine similarity | `cosine_similarity_matching_external_reference.py` | `distributional_check_external_references_cosine.py` |
| Soft-n-gram matching | `soft_n_gram_matching_external_reference.py` | `distributional_check_external_references.py` |
| Soft-keypoint matching  | `soft_keypoint_matching_external_reference.py` | `distributional_check_external_references.py` |

```bash
python <run_script> \
	--embedding_model [EMBEDDING_MODEL] \
	--batch_size [BATCH_SIZE] \
	--all_files_path [PATHFILE] \
	--dataset_subset_path hard-subset \
	--out_suffix [SUFFIX] \
	--similarity_threshold [THRESHOLDS] \ # only for soft-n-gram and soft-keypoint matching
	--min_n [INT] \ # only for soft-n-gram matching
	--max_n [INT, >= min_n] # only for soft-n-gram matching
```
The scripts support the following choices of `[EMBEDDING_MODEL]`: `linq-embed-mistral`, `specter2` and `text-embedding-3-(small/large)` (in which case `OPENAI_API_KEY` must be set). It adaptively adjusts the batch size to optimally use space available in a A600 (48G), starting from an initial guess of `[BATCH_SIZE]` and reducing it until it fits. `[PATHFILE]` is the text file with paths of all reviews for which scores have to be computed (reference reviews are located automatically). Soft-n-gram and soft-keypoint matching require an additional threshold parameter: `[THRESHOLD]` can either be a float or a comma-separated list of floats. In the later case, scores for all the thresholds in the list are computed and saved. Soft-n-gram matching considers n-grams of size in the range `[min_n,max_n]` (for reproducing our results, set both to 40). 

The computed similarity scores are saved in `results/SIMILARITY_METRIC/hard-subset/EMBEDDING_MODEL/result_anchor_features_[SUFFIX].json`.

To visualize these similarity scores, run

```bash
python <analyse_script> \
	--results_file results/SIMILARITY_METRIC/hard-subset/EMBEDDING_MODEL/result_anchor_features_[SUFFIX].json \
	--threshold [FLOAT] \
	--score_aggregation_mode [max/avg] \
	--output_dir [DESTINATION_DIR] \
	--monochrome \
	--vertical
```

To train a classifier with these similarity scores, run `peer-review-advantages/Anchor/ai-reference-similarity-classifier.ipynb` after replacing `[SIMILARITY_FEATURES_SOURCE]` by `results/SIMILARITY_METRIC/hard-subset/EMBEDDING_MODEL/result_anchor_features_[SUFFIX].json` (and setting a threshold of choice, except for cosine similarity).

### Stylometric/linguistic features

We choose a set of $38$ such features capturing lexical diversity (e.g. type-token ratio, bigram, trigram uniqueness, etc.), POS tags (e.g. verb percentage, abstract noun percentage, etc.), readability (e.g. average syllables per word, Flesch reading ease, etc.) and other statistics. Refer to the `peer-review-advantages/linguistic_features` directory.

First, save the linguistic features of reviews listed in a `[PATHFILE]` to `[DESTINATION_DIR]/hard-subset/linguistic_features_[SUFFIX].json` by running:

```bash
python save_linguistic_features_fast.py \
	--pathfile [PATHFILE] \
	--source_dir hard-subset \
	--output_dir [DESTINATION_DIR] \
	--suffix [SUFFIX]
```

To train a classifier with these similarity scores, run `peer-review-advantages/linguistic_features/linguistic-features-classifier.ipynb` after replacing `[LINGUISTIC_FEATURES_SOURCE]` by `[DESTINATION_DIR]/hard-subset/linguistic_features_[SUFFIX].json`.

### RoBERTa classifier

We do a full finetuning of a RoBERTa model with a $5$-label classification head. We train the classifier on overlapping segments of fixed length from review texts. Final prediction is obtained through majority voting over segment-level predictions. Refer to `peer-review-advantages/roberta-classifier`

First, save the fragmented segments as separate examples with corresponding labels in a CSV file with:

```bash
python prepare_data_simple.py \
	--path_file [PATHFILE] \
	--fragment [FRAGMENT_LEN_IN_CHARS] \
	--overlap [OVERLAP_LEN_IN_CHARS] \
	--output_file [CSV_PATH]
```

Then, train the RoBERTa classifier with the following:

```bash
python train-roberta-classifier.py \
	--source_csv [CSV_PATH] \
	--leave_model [MODEL_ID]
```

The `--leave_model` argument is optional. If set, it removes examples from `[MODEL_ID]` from training and evaluates only on examples from this model.  