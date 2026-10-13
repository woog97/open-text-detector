# Overview of the database: `database.db`

This database is seperated into the following tables:

* `answers`: All written and generated essays.
* `datasets`: Dataset name and description.
* `experiments`: Information about the detector experiments.
* `human_meta`: Metadata about human written essays (if exist).
* `jobs`: Information about the LLM jobs to generate essays.
* `predictions`: Detector predictions for each essay from the `answers` table.
* `questions`: Task descriptions, the essays are based on.
* `summaries`: Summaries of human-written texts.

## Table: `answers`

| Column        | Type      | NOT NULL | Default           | Primary Key |
|---------------|-----------|----------|-------------------|-------------|
| `id`          | INTEGER   | False    |                   | True        |
| `question_id` | INTEGER   | True     |                   | False       |
| `job_id`      | INTEGER   | False    |                   | False       |
| `is_human`    | BOOLEAN   | True     | FALSE             | False       |
| `answer`      | TEXT      | True     |                   | False       |
| `rewrite_of`  | TEXT      | False    |                   | False       |
| `created_at`  | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at` | TIMESTAMP | False    |                   | False       |

## Table: `datasets`

| Column        | Type      | NOT NULL | Default           | Primary Key |
|---------------|-----------|----------|-------------------|-------------|
| `id`          | INTEGER   | False    |                   | True        |
| `name`        | TEXT      | True     |                   | False       |
| `description` | TEXT      | False    |                   | False       |
| `created_at`  | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at` | TIMESTAMP | False    |                   | False       |

## Table: `experiments`

| Column             | Type      | NOT NULL | Default           | Primary Key |
|--------------------|-----------|----------|-------------------|-------------|
| `id`               | INTEGER   | False    |                   | True        |
| `job_id`           | INTEGER   | True     |                   | False       |
| `dataset_id`       | INTEGER   | True     |                   | False       |
| `text_author`      | TEXT      | True     |                   | False       |
| `prompt_mode`      | TEXT      | True     |                   | False       |
| `model`            | TEXT      | True     |                   | False       |
| `n_samples`        | INTEGER   | True     |                   | False       |
| `max_words`        | INTEGER   | True     |                   | False       |
| `cut_sentences`    | BOOLEAN   | False    | FALSE             | False       |
| `use_cache`        | BOOLEAN   | False    | FALSE             | False       |
| `human_hash`       | TEXT      | False    |                   | False       |
| `llm_hash`         | TEXT      | False    |                   | False       |
| `seed`             | INTEGER   | True     |                   | False       |
| `execution_time`   | INTEGER   | False    |                   | False       |
| `model_checkpoint` | TEXT      | False    |                   | False       |
| `created_at`       | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`      | TIMESTAMP | False    |                   | False       |

## Table: `human_meta`

| Column                       | Type      | NOT NULL | Default           | Primary Key |
|------------------------------|-----------|----------|-------------------|-------------|
| `answer_id`                  | INTEGER   | False    |                   | True        |
| `level`                      | FLOAT     | False    |                   | False       |
| `grade`                      | TEXT      | False    |                   | False       |
| `gender`                     | TEXT      | False    |                   | False       |
| `year_of_birth`              | INTEGER   | False    |                   | False       |
| `L1`                         | TEXT      | False    |                   | False       |
| `ethnicity`                  | TEXT      | False    |                   | False       |
| `economically_disadvantaged` | TEXT      | False    |                   | False       |
| `student_disability_status`  | TEXT      | False    |                   | False       |
| `created_at`                 | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`                | TIMESTAMP | False    |                   | False       |

## Table: `jobs`

| Column           | Type      | NOT NULL | Default           | Primary Key |
|------------------|-----------|----------|-------------------|-------------|
| `id`             | INTEGER   | False    |                   | True        |
| `instruction`    | TEXT      | False    |                   | False       |
| `prompt`         | TEXT      | False    |                   | False       |
| `model`          | TEXT      | False    |                   | False       |
| `prompt_mode`    | TEXT      | False    |                   | False       |
| `summary_file`   | TEXT      | False    |                   | False       |
| `dataset_id`     | INTEGER   | False    |                   | False       |
| `temperature`    | FLOAT     | False    |                   | False       |
| `max_new_tokens` | INTEGER   | False    |                   | False       |
| `batch_size`     | INTEGER   | False    |                   | False       |
| `created_at`     | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`    | TIMESTAMP | False    |                   | False       |
| `status`         | TEXT      | False    | "running"         | False       |
| `batch_id`       | TEXT      | False    | NULL              | False       |

## Table: `predictions`

| Column          | Type      | NOT NULL | Default           | Primary Key |
|-----------------|-----------|----------|-------------------|-------------|
| `id`            | INTEGER   | False    |                   | True        |
| `experiment_id` | INTEGER   | True     |                   | False       |
| `answer_id`     | INTEGER   | True     |                   | False       |
| `prediction`    | TEXT      | True     |                   | False       |
| `created_at`    | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`   | TIMESTAMP | False    |                   | False       |

## Table: `questions`

| Column                   | Type      | NOT NULL | Default           | Primary Key |
|--------------------------|-----------|----------|-------------------|-------------|
| `id`                     | INTEGER   | False    |                   | True        |
| `dataset_id`             | INTEGER   | True     |                   | False       |
| `question`               | TEXT      | True     |                   | False       |
| `is_original`            | BOOLEAN   | True     | TRUE              | False       |
| `rewrite_from`           | INTEGER   | False    |                   | False       |
| `created_at`             | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`            | TIMESTAMP | False    |                   | False       |
| `module`                 | TEXT      | False    |                   | False       |
| `discipline`             | TEXT      | False    |                   | False       |
| `discipline_group`       | TEXT      | False    |                   | False       |
| `course`                 | TEXT      | False    |                   | False       |
| `task_Typee`             | TEXT      | False    |                   | False       |
| `supplementary_material` | TEXT      | False    |                   | False       |

## Table: `summaries`

| Column           | Type      | NOT NULL | Default           | Primary Key |
|------------------|-----------|----------|-------------------|-------------|
| `id`             | INTEGER   | False    |                   | True        |
| `source_id`      | INTEGER   | True     |                   | False       |
| `summary`        | TEXT      | True     |                   | False       |
| `model`          | TEXT      | True     |                   | False       |
| `max_new_tokens` | INTEGER   | True     |                   | False       |
| `created_at`     | TIMESTAMP | False    | CURRENT_TIMESTAMP | False       |
| `modified_at`    | TIMESTAMP | False    |                   | False       |
