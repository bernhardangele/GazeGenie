# Single-file workflows

The single-file workflows are the best place to tune settings before running batch jobs.

## A. Single `.asc` workflow

Open **Single File 📁** and then the **`.asc files`** tab.

### Step 1: Optional configuration reload

If you already saved a working ASC configuration:

1. Expand **Load config file**.
2. Upload the saved `.json` file.
3. Click **Load in config**.

This restores the current ASC-related parsing, cleaning, correction, and analysis settings.

### Step 2: Choose the input file

You can either:

- upload one `.asc` file,
- upload associated `.ias` files if stimulus data is not embedded in the ASC,
- or switch to an included example file.

Use `.ias` files when the ASC references stimulus files with `IAREA FILE` markers.

### Step 3: Set file parsing options

Before loading the file, review these settings:

- trial start keyword,
- custom trial start keyword,
- trial end keyword,
- custom trial end keyword,
- whether gaps between words should be closed,
- whether gaps between lines should be closed,
- whether practice/question trials should be filtered out.

These settings control how GazeGenie finds trials and reconstructs the stimulus geometry.

### Step 4: Load the ASC data

1. Click **Load selected data.**
2. Review the metadata shown under **Metadata found in .asc file**.
3. Confirm that trial identifiers, stimulus information, and trial counts look sensible.

If the metadata is wrong, fix the parsing settings before moving on.

### Step 5: Select and load one trial

1. In **Trial and algorithm selection**, choose a trial.
2. Decide whether fixations that begin before the detected start of the trial but end after it should be discarded.
3. Click **Load trial**.

After loading, inspect:

- **Show Trial Information**,
- **Show fixations and saccades before cleaning**.

This is your baseline before any cleaning or line assignment is applied.

### Step 6: Clean the fixation sequence

Use **Cleaning options** and start with the defaults recommended by the manuscript:

- discard blink-adjacent fixations,
- discard far-outside-text fixations,
- horizontal threshold: `2.0` character widths,
- vertical threshold: `0.5` line heights,
- discard long fixations over `800 ms`,
- short-fixation handling: **Merge then discard**,
- short-fixation threshold: `80 ms`,
- merge distance threshold: `1` character width.

Then click **Apply cleaning**.

Inspect the cleaning results before continuing. The manuscript's advice still applies: tune these values on several representative trials, not just one.

### Step 7: Choose line-assignment algorithms

Once cleaning looks reasonable:

1. Choose one or more line-assignment algorithms.
2. Choose fixation-level features to calculate.
3. Click the correction button for the trial.

The default algorithm pair is:

- `slice`
- `DIST`

A good practical strategy is to compare at least one classical algorithm with one DIST-based method.

### Step 8: Inspect corrected outputs

After correction, review:

- corrected fixation dataframe,
- saccade dataframe,
- stimulus dataframe,
- corrected fixation plots,
- y-correction plots,
- optional fixation/saccade feature plots.

You can choose what appears in the main plot, including:

- uncorrected fixations,
- corrected fixations,
- word boxes,
- characters,
- character boxes.

### Step 9: Run analysis

The analysis section supports two analysis styles.

#### Analysis without eyekit

Use this when the automatically reconstructed stimulus coordinates are already correct.

1. Choose which corrected algorithm result should drive the analysis.
2. Select the word measures to calculate.
3. Select the sentence measures to calculate.
4. Decide whether to include word bounding-box coordinates in the output.
5. Inspect the word and sentence tables.
6. Pick one computed measure to visualize on the stimulus.

#### Analysis using eyekit

Use this when you want manual control over stimulus alignment.

1. Open the eyekit tab.
2. Choose the corrected algorithm to use.
3. Adjust font, font size, x position, y position, and line height.
4. Apply the selected parameters.
5. Inspect the eyekit visualization.
6. Download eyekit fixation JSON, eyekit textblock JSON, and eyekit word-measure CSVs if needed.

### Step 10: Save the working configuration

When you are satisfied with the ASC setup, download the configuration JSON using the single-file ASC settings download button. Reuse that file in the multiple-file ASC workflow.

## B. Single custom-file workflow

Open **Single File 📁** and then **custom files**.

This workflow is for fixation tables exported as `.csv`, `.txt`, or `.dat` plus a separate stimulus source.

### Supported stimulus inputs

The stimulus file can currently be:

- `.json`,
- `.csv`, `.txt`, or `.dat`,
- `.png` or `.jpeg`/`.jpg`.

If you upload an image, GazeGenie runs OCR to infer character boxes.

### Step 1: Load the fixation and stimulus files

1. Upload the fixation file.
2. Upload the stimulus file.
3. Set whether word gaps and line gaps should be closed.
4. Choose uploaded files or the bundled example files.
5. Click **Load selected data.**

### Step 2: Preview the loaded files

Use **Preview loaded files** to confirm that:

- the fixation table loaded correctly,
- the stimulus file loaded correctly,
- trial identifiers look consistent.

### Step 3: Confirm column or key mappings

Open **Column names for csv files** and verify the mappings for:

- fixation x/y,
- subject id,
- trial id,
- fixation start time,
- fixation end time,
- stimulus x/y center,
- stimulus x/y min/max,
- stimulus character content,
- line number,
- stimulus trial id.

Then click **Confirm column/key names**.

This is the most important step for custom inputs. If the mappings are wrong, later cleaning, correction, and analysis will also be wrong.

### Step 4: Select a trial

If the files contain multiple trials:

1. Choose the desired trial.
2. Click **Select trial**.

### Step 5: Clean fixations

The custom-file workflow uses the same cleaning settings as the ASC workflow:

- blink-adjacent filtering,
- outside-text filtering,
- long-fixation filtering,
- short-fixation handling,
- distance thresholds.

Apply cleaning and inspect the cleaning results.

### Step 6: Correct fixations

1. Select one or more line-assignment algorithms.
2. Click **Correct fixations**.
3. Download the corrected fixation CSV if needed.

### Step 7: Inspect plots and analysis

You can then:

- view corrected fixation data,
- choose plot contents,
- run analysis without eyekit,
- or run analysis using eyekit.

### Important notes for image-based stimulus extraction

If you upload an image as the stimulus source:

- the filename stem **must match the trial id**,
- OCR is used to create character boxes,
- the OCR-derived stimulus table is saved into `results/`,
- repeated runs can reuse cached OCR results from `results/ocr_cache`.
