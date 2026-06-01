#set page(
  width: 841mm,
  height: 594mm,
  margin: 16mm,
)

#set text(size: 14pt)
#set heading(numbering: none)
#set par(justify: false)

#let card(fill-color, title, body, img: none) = rect(
  fill: fill-color,
  radius: 6pt,
  inset: 10pt,
  stroke: rgb("#2f3b4a"),
  width: 100%,
)[
  #text(18pt, weight: "bold")[#title]
  #v(4pt)
  #body
  #if img != none {
    v(6pt)
    image(img, width: 100%)
  }
]

#align(center)[
  #text(42pt, weight: "bold")[GazeGenie Workflow]
  #v(4pt)
  #text(22pt)[From raw reading eye-tracking data to robust, reproducible measures — at any scale]
]

#v(10pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  [
    #card(rgb("#e8f1ff"), [1) Input & Setup], [
      - Eyelink `.asc` *or* pre-extracted fixation `.csv/.txt/.dat` — no reformatting needed.
      - Stimulus from embedded ASC data, character tables, JSON, or raw images (OCR auto-extracts bounding boxes).
      - Zero-install browser launch via Codespaces or Docker; also runs fully local.
      - Reusable JSON configs guarantee identical settings across sessions and collaborators.
    ], img: "../manuscript/ims/file_up_single_asc.png")
  ],
  [
    #card(rgb("#eef9f0"), [2) Parse & Validate], [
      - Flexible trial detection: pick from common start/end flags or define your own.
      - Auto-extracts per-trial metadata (ID, condition, item, screen size, timestamps).
      - Optional exclusion of practice and question trials keeps analysis clean.
      - Instant validation view lets you catch mis-parsed trials before committing.
    ], img: "../manuscript/ims/col_names_choice_csv.png")
  ],
  [
    #card(rgb("#fff6e9"), [3) Clean], [
      - Removes blink-adjacent fixations and those outside the stimulus region.
      - Configurable horizontal/vertical margins adapt to any stimulus geometry.
      - Smart short-fixation handling: merge, discard, or merge-then-discard with one click.
      - Color-coded cleaning visualization reveals exactly which fixations are affected and why.
    ], img: "../manuscript/ims/clean_res.png")
  ],

  [
    #card(rgb("#efeafe"), [4) Correct Line Assignment], [
      - 15+ classical and deep-learning algorithms (incl. DIST, Wisdom of Crowds, DIST-Ensemble).
      - Run and *visually compare* multiple algorithms side-by-side on the same trial.
      - Interactive fixation plots and per-algorithm y-correction summary catch outliers instantly.
      - Export corrected fixations and re-aligned saccades ready for downstream analysis.
    ], img: "../manuscript/ims/fix_corr_fix_plot.png")
  ],
  [
    #card(rgb("#e9fbfb"), [5) Analyze], [
      - Rich fixation-, word-, and sentence-level measure library (FFD, GD, GPT, skipping, regressions, …).
      - Word-level heat-map overlaid on the stimulus makes results immediately interpretable.
      - Eyekit-compatible JSON export for reproducible, shareable alignment pipelines.
      - Interactive feature plots for fixations and saccades support rapid quality checks.
    ], img: "../manuscript/ims/word_measures.png")
  ],
  [
    #card(rgb("#f3f3f3"), [6) Scale to Batch Jobs], [
      - Lock tuned single-trial settings into a JSON config and apply to hundreds of files.
      - Optional multiprocessing slashes wall-clock time on large corpora.
      - Per-trial plots and data files saved automatically; consolidated ZIP download in one click.
      - Post-batch trial drill-down lets you inspect any individual trial without re-running.
    ], img: "../manuscript/ims/fix_sacc_feat_plots.png")
  ],
)

#v(10pt)

#grid(
  columns: (3fr, 2fr),
  gutter: 12pt,
  [
    #card(rgb("#dde9ff"), [Recommended Operating Pattern], [
      1. Tune cleaning and correction parameters on representative single trials.
      2. Compare multiple correction algorithms and select the best fit for your data.
      3. Lock parameters and save the full configuration to a JSON file.
      4. Load the config into the Batch tab and launch large-scale processing.
      5. Audit concatenated outputs at subject, trial, word, and sentence level.
      6. Use post-batch trial drill-down for targeted quality assurance.
    ])
  ],
  [
    #card(rgb("#ffe8e8"), [Why GazeGenie?], [
      - *Single unified interface* from raw ASC parsing all the way to publication-ready measures.
      - *OCR-powered* stimulus extraction works even when no bounding-box file is available.
      - *15+ correction algorithms* compared side-by-side — no separate tooling needed.
      - *Batch-ready by design*: one JSON config reproduces every processing decision at scale.
      - *Interactive visualizations* at every step make errors visible before they propagate.
      - *Eyekit integration* with JSON export ensures full reproducibility and cross-tool compatibility.
    ])
  ],
)

#v(10pt)

#align(center)[
  #text(16pt, weight: "bold")[Landscape DIN A1 — GazeGenie · github.com/bernhardangele/GazeGenie]
]
