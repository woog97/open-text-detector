---
pretty_name: Open Text Detector Paper Text
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---
# Open Text Detector Paper Text

14,561 papers, one row per PDF, with canonical extracted text and provenance. This private dataset belongs to **open-text-detector** and is available to organization members. The original private [`woog/open-text-detector-paper-text`](https://huggingface.co/datasets/woog/open-text-detector-paper-text) is preserved separately.

## Browse samples

[Open the interactive table of all samples and columns](https://huggingface.co/datasets/open-text-detector/paper-text/viewer/default/train). Sign in with your organization account. The Dataset Viewer lets you browse records and inspect each column; Parquet is the underlying storage format and does not require downloading files to browse the table.

| Column | Contents |
|---|---|
| `paper_id` | Paper identifier |
| `forum_id` | Source forum identifier |
| `title` | Paper title |
| `conference` | Conference/source venue |
| `year` | Recorded year |
| `text` | Full extracted paper text |
| `pdf_sha256` | Source PDF checksum |
| `text_sha256` | Extracted text checksum |
| `extraction_sha256` | Extraction checksum |
| `extraction_method` | Recorded extraction method |
| `extraction_metadata_json` | Extraction metadata and caveats, encoded as JSON |
| `page_count` | Recorded PDF page count |

## Provenance and use

The Parquet file is byte-identical to the source release at revision `a79fcf1ffcb45a20bf6c9728142d0fa974304e95`; SHA256: `2fb3ec7e1139808c0f3eea300d8f1a3d78b9e0ee11291ead7bdf020d6dbf1fcf`.

`train` contains the full corpus and is a storage convention, not a curated train/evaluation split. Paper text is not labeled as human-only authorship. PDF/text hashes identify exact versions; metadata records extraction methods and caveats. PDFs, position rectangles and classifier predictions are excluded. Pin a dataset commit for reproducible runs.
