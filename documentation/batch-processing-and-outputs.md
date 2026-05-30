# Batch processing and outputs

Batch processing should usually be done only after you have already validated settings in the single-file workflow.

## A. Multiple ASC files

Open **Multiple Files 📁 📁**.

### Step 1: Optional configuration reload

If you previously saved a good ASC configuration JSON, load it first from **Load config file.**

### Step 2: Upload files

In **Upload files and choose configuration options.** you can upload:

- multiple `.asc` files,
- a `.zip` or `.tar` archive containing ASC files,
- matching `.ias` files when the stimuli are not embedded.

### Step 3: Set parsing and filtering options

Review the same parsing settings used in the single-file ASC workflow:

- trial start/end keywords,
- custom keyword fields,
- gap-closing options,
- practice/question filtering.

### Step 4: Set cleaning options

The batch ASC workflow exposes the same cleaning controls as the single-file workflow, including:

- trial-start overlap handling,
- blink-adjacent filtering,
- outside-text filtering,
- horizontal and vertical thresholds,
- long-fixation filtering and threshold,
- short-fixation handling,
- short-fixation threshold,
- merge-distance threshold.

### Step 5: Set correction and analysis options

Choose:

- one or more line-assignment algorithms,
- fixation-level measures,
- word-level measures,
- whether to include word coordinates,
- sentence-level measures.

You can also enable:

- **Process trials in parallel (fast but experimental)**,
- **Save fixations, saccades, stimulus and metadata for each trial to a separate file**.

### Step 6: Run the batch job

Click **🚀 Process files**.

Depending on file count, trial count, and algorithm selection, this can take time.

### Step 7: Review batch outputs

After processing, inspect the generated sections, including:

- metadata by subject and trial,
- item-level stimulus overview,
- subject-level summary statistics,
- trial-level summary statistics,
- combined fixations dataframe,
- combined saccades dataframe,
- combined words dataframe,
- combined sentence dataframe.

### Step 8: Download results

Use the zip selector and download button to retrieve the batch output archive.

### Step 9: Inspect individual processed trials

The current interface also lets you select one processed trial and then:

- review trial information,
- inspect cleaned fixations,
- inspect corrected fixations and saccades,
- inspect the stimulus dataframe,
- view corrected plots,
- run per-trial analysis without eyekit,
- run per-trial analysis with eyekit.

This post-batch drill-down workflow is one of the useful additions beyond the manuscript draft.

## B. Multiple fixation CSV files plus images

The application now also supports bulk processing of custom fixation files.

Open **Upload multiple CSV files and image files for bulk processing.**

### Required inputs

Upload:

- one or more fixation files (`.csv`, `.txt`, `.dat`),
- one or more stimulus images (`.png`, `.jpg`, `.jpeg`).

The image filename stem must match the `trial_id` found in the fixation table.

### Configuration

The batch CSV workflow lets you configure:

- whether word and line gaps are closed,
- cleaning options,
- line-assignment algorithms,
- fixation-level measures,
- word-level measures,
- sentence-level measures,
- whether word coordinates should be included,
- whether per-trial files should be saved individually.

### Run and review

1. Click **🚀 Process CSV files**.
2. Review subject/trial metadata and summary tables.
3. Inspect combined fixations, word, and sentence dataframes.
4. Download the generated zip file.
5. Use the individual-trial inspection section to review one processed trial in detail.

### How batch CSV processing works

For each `(subject, trial_id)` combination in the uploaded fixation tables, GazeGenie:

1. finds the matching image by `trial_id`,
2. runs or reuses OCR,
3. constructs character and word geometry,
4. cleans the fixation data,
5. applies the selected line-assignment algorithms,
6. computes the requested measures,
7. writes combined and downloadable outputs.

Trials without matching images are skipped and warned about in the UI.

## Output types you should expect

Depending on workflow and settings, GazeGenie may produce:

- corrected fixation CSVs,
- saccade tables,
- stimulus/character tables,
- word-level measure tables,
- sentence-level measure tables,
- summary statistics by trial and subject,
- per-trial metadata,
- plots,
- zipped result bundles,
- OCR-derived stimulus CSV files,
- eyekit JSON exports for inspected trials.

## Practical advice for batch jobs

- Tune settings on several representative trials before scaling up.
- Compare at least two line-assignment algorithms before choosing one final analysis path.
- Be conservative with outside-text thresholds so you do not remove legitimate reading fixations.
- Use multiprocessing only after a small validation run succeeds.
- Use per-trial file saving only when you really need it, because it creates many files.
