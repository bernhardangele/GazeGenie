# GazeGenie documentation

This folder contains a practical user guide for the current GazeGenie interface.

GazeGenie is a Streamlit application for parsing, cleaning, correcting, visualizing, and analyzing fixation data from reading experiments. The manuscript in `/tmp/workspace/bernhardangele/GazeGenie/manuscript` describes the general workflow well, but the application has grown since that draft was written. These documents combine:

- the recommended workflow from the manuscript,
- the current installation and launch options from the repository,
- and the current user-facing options exposed in the Python code.

## Start here

1. Read [`installation-and-launch.md`](installation-and-launch.md) to choose how you want to run the tool.
2. If you are working with Eyelink `.asc` files, follow [`single-file-workflows.md`](single-file-workflows.md) first and tune settings on one trial.
3. If you are working with exported fixation tables plus stimulus files or images, use [`single-file-workflows.md`](single-file-workflows.md) for the custom-file workflow.
4. Once your settings work on representative trials, move to [`batch-processing-and-outputs.md`](batch-processing-and-outputs.md).
5. Use [`options-reference.md`](options-reference.md) whenever you need the exact current option names and supported values.

## Recommended workflow

The current best-practice workflow is still the one described in the manuscript:

1. Open a single file first.
2. Confirm that trials and metadata were parsed correctly.
3. Select one representative trial.
4. Tune cleaning settings.
5. Compare multiple line-assignment algorithms.
6. Inspect corrected plots and feature tables.
7. Choose the word and sentence measures you need.
8. Save the configuration as JSON.
9. Reuse that configuration in the multiple-file workflow.
10. Inspect the combined outputs and download the final zip file.

## Important features added since the manuscript draft

The current application includes several newer workflow features that should be considered part of normal use:

- **GitHub Codespaces support** for browser-based use without local setup.
- **Custom fixation-file workflows** for CSV/TXT/DAT inputs.
- **Stimulus extraction from images** (`.png`, `.jpg`, `.jpeg`) via OCR for custom-file workflows.
- **Batch CSV processing with matching stimulus images**.
- **OCR result caching** in `results/ocr_cache` to speed up repeated image-based runs.
- **Reloadable JSON configuration files** for single and multi-file ASC workflows.
- **Eyekit-based analysis and export**, including downloadable eyekit fixation and textblock JSON files.
- **Interactive post-batch trial inspection** for both ASC and CSV bulk runs.

## Related reference material already in the repository

The application also ships with column-definition reference files that are useful when interpreting outputs:

- `/tmp/workspace/bernhardangele/GazeGenie/fixations_df_columns.md`
- `/tmp/workspace/bernhardangele/GazeGenie/saccades_df_columns.md`
- `/tmp/workspace/bernhardangele/GazeGenie/trials_df_columns.md`
- `/tmp/workspace/bernhardangele/GazeGenie/item_df_columns.md`
- `/tmp/workspace/bernhardangele/GazeGenie/word_measures.md`
- `/tmp/workspace/bernhardangele/GazeGenie/sentence_measures.md`
- `/tmp/workspace/bernhardangele/GazeGenie/subject_measures.md`
- `/tmp/workspace/bernhardangele/GazeGenie/chars_df_columns.md`
