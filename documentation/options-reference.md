# Options reference

This file lists the main current user-facing options exposed by the app.

## 1. ASC parsing options

### Trial start keyword options

- `SYNCTIME`
- `START`
- `GAZE TARGET ON`
- `custom`

### Trial end keyword options

- `ENDBUTTON`
- `END`
- `KEYBOARD`
- `custom`

### Additional parsing/filtering controls

- **Custom trial start keyword**
- **Custom trial end keyword**
- **Should spaces between words be included in word bounding box?**
- **Should spaces between lines be included in word and character bounding boxes?**
- **Should Practice and question trials be excluded if possible?**

## 2. Cleaning options

### Short-fixation handling modes

- `Merge`
- `Merge then discard`
- `Discard`
- `Leave unchanged`

### Default cleaning values used in the UI

- long-fixation threshold: `800 ms`
- short-fixation threshold: `80 ms`
- merge distance threshold: `1` character width
- outside-text horizontal threshold: `2.0` character widths
- outside-text vertical threshold: `0.5` line heights

### Cleaning controls available in the UI

- discard fixations that overlap the detected trial start
- discard fixations just before or after blinks
- discard fixations far outside the text
- set maximum horizontal distance from the text
- set maximum vertical distance from the text
- discard long fixations
- set maximum fixation duration
- choose how short fixations are handled
- set minimum fixation duration
- set maximum merge distance

## 3. Plot options

### Main fixation/stimulus plot layers

- `Uncorrected Fixations`
- `Corrected Fixations`
- `Word boxes`
- `Characters`
- `Character boxes`

The plotting sections also support:

- choosing the display font,
- showing or hiding plots,
- visualizing selected fixation features,
- visualizing selected saccade features,
- choosing `Index` or `Start Time` as the feature-plot x-axis.

## 4. Line-assignment algorithms currently exposed

- `warp`
- `regress`
- `compare`
- `attach`
- `segment`
- `split`
- `stretch`
- `chain`
- `slice`
- `cluster`
- `merge`
- `Wisdom_of_Crowds`
- `DIST`
- `DIST-Ensemble`
- `Wisdom_of_Crowds_with_DIST`
- `Wisdom_of_Crowds_with_DIST_Ensemble`

### Default selected algorithms

- `slice`
- `DIST`

## 5. Fixation-level measures

### Default fixation measures

- `letternum`
- `letter`
- `on_word_number`
- `on_word`
- `on_sentence`
- `num_words_in_sentence`
- `on_sentence_num`
- `word_land`
- `line_let`
- `line_word`
- `sac_in`
- `sac_out`
- `word_launch`
- `word_refix`
- `word_reg_in`
- `word_reg_out`
- `sentence_reg_in`
- `word_firstskip`
- `word_run`
- `sentence_run`
- `word_run_fix`
- `word_cland`

### Additional fixation measures exposed in the app

- `angle_incoming`
- `angle_outgoing`
- `line_let_from_last_letter`
- `sentence_word`
- `line_let_previous`
- `line_let_next`
- `sentence_refix`
- `word_reg_out_to`
- `word_reg_in_from`
- `sentence_reg_out`
- `sentence_reg_in_from`
- `sentence_reg_out_to`
- `sentence_firstskip`
- `word_runid`
- `sentence_runid`
- `word_fix`
- `sentence_fix`
- `sentence_run_fix`

## 6. Word-level measures currently exposed

### Default word measures

- `firstrun_dur`
- `firstrun_nfix`
- `firstfix_dur`
- `singlefix_dur`
- `total_fixation_duration`
- `firstrun_gopast`
- `skip`
- `reg_in`
- `reg_out`
- `number_of_fixations`
- `number_of_regressions_in`

### Full word-measure list

- `blink`
- `first_of_many_duration`
- `firstfix_cland`
- `firstfix_dur`
- `firstfix_land`
- `firstfix_launch`
- `firstfix_sac_in`
- `firstfix_sac_out`
- `firstrun_blink`
- `firstrun_dur`
- `firstrun_gopast`
- `firstrun_gopast_sel`
- `firstrun_nfix`
- `firstrun_refix`
- `firstrun_reg_in`
- `firstrun_reg_out`
- `firstrun_skip`
- `gopast`
- `gopast_sel`
- `initial_landing_distance`
- `initial_landing_position`
- `landing_distances`
- `nrun`
- `number_of_fixations`
- `number_of_regressions_in`
- `refix`
- `skip`
- `reg_in`
- `reg_out`
- `reread`
- `second_pass_duration`
- `singlefix`
- `singlefix_cland`
- `singlefix_dur`
- `singlefix_land`
- `singlefix_launch`
- `singlefix_sac_in`
- `singlefix_sac_out`
- `total_fixation_duration`

## 7. Sentence-level measures currently exposed

### Default sentence measures

- `on_sentence_num`
- `on_sentence`
- `num_words_in_sentence`
- `total_n_fixations`
- `total_dur`

### Full sentence-measure list

- `on_sentence_num`
- `on_sentence`
- `num_words_in_sentence`
- `skip`
- `nrun`
- `reread`
- `reg_in`
- `reg_out`
- `total_n_fixations`
- `total_dur`
- `rate`
- `gopast`
- `gopast_sel`
- `firstrun_skip`
- `firstrun_reg_in`
- `firstrun_reg_out`
- `firstpass_n_fixations`
- `firstpass_dur`
- `firstpass_forward_n_fixations`
- `firstpass_forward_dur`
- `firstpass_reread_n_fixations`
- `firstpass_reread_dur`
- `lookback_n_fixations`
- `lookback_dur`
- `lookfrom_n_fixations`
- `lookfrom_dur`

## 8. Custom-file mapping expectations

### Fixation-file fields expected by the workflow

The custom-file workflow expects mappings for:

- x coordinate
- y coordinate
- subject id
- trial id
- fixation start time
- fixation end time

### Stimulus-file fields expected by the workflow

The custom-file workflow expects mappings for:

- character x center
- character y center
- character x min
- character x max
- character y min
- character y max
- character content
- assigned line number
- trial id

## 9. Eyekit controls

When you use eyekit-based analysis, the current UI lets you set:

- font face
- font size
- x position of the first character
- y position of the first character
- line height

You can enter these values either with:

- **Sliders**, or
- **Direct input**.

## 10. Current features that were not central in the manuscript draft

The current codebase exposes the following workflow features that should be treated as part of the supported interface:

- JSON config download and reload for ASC workflows
- custom fixation-file workflows
- OCR-based stimulus extraction from images
- cached OCR results
- batch CSV-plus-image processing
- trial inspection after batch runs
- eyekit JSON export buttons
- optional saving of per-trial files during batch processing
