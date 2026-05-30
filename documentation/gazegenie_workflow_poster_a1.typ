#set page(
  width: 841mm,
  height: 594mm,
  margin: 16mm,
)

#set text(size: 14pt)
#set heading(numbering: none)
#set par(justify: false)

#let card(fill-color, title, body) = rect(
  fill: fill-color,
  radius: 6pt,
  inset: 10pt,
  stroke: rgb("#2f3b4a"),
)[
  #text(18pt, weight: "bold")[#title]
  #v(4pt)
  #body
]

#align(center)[
  #text(42pt, weight: "bold")[GazeGenie Workflow]
  #v(4pt)
  #text(22pt)[From raw reading eye-tracking data to robust measures at scale]
]

#v(10pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  [
    #card(rgb("#e8f1ff"), [1) Input & Setup], [
      - Supported sources: Eyelink `.asc` or custom fixation `.csv/.txt/.dat`.
      - Stimulus from embedded data, text tables, JSON, or images.
      - Browser-first launch options (Codespaces, Docker, local).
      - Reusable JSON configs to standardize workflows.
    ])
  ],
  [
    #card(rgb("#eef9f0"), [2) Parse & Validate], [
      - Detect trials and metadata with configurable start/end keywords.
      - Optional filtering of practice/question trials.
      - Validate parsed trial structure before processing.
      - Confirm custom column mappings for imported fixation tables.
    ])
  ],
  [
    #card(rgb("#fff6e9"), [3) Clean], [
      - Remove blink-adjacent and outside-text fixations.
      - Tune horizontal/vertical thresholds by stimulus geometry.
      - Handle short and long fixations with configurable policies.
      - Inspect cleaned trajectories before correction.
    ])
  ],

  [
    #card(rgb("#efeafe"), [4) Correct Line Assignment], [
      - Compare classical and DIST-based algorithms.
      - Defaults (`slice` + `DIST`) provide strong baseline behavior.
      - Visualize y-correction and fixation path changes.
      - Export corrected fixations and saccades.
    ])
  ],
  [
    #card(rgb("#e9fbfb"), [5) Analyze], [
      - Compute fixation-, word-, and sentence-level measures.
      - Run direct analysis or eyekit-aligned analysis.
      - Control exported coordinate detail.
      - Inspect outcomes with interactive plots and tables.
    ])
  ],
  [
    #card(rgb("#f3f3f3"), [6) Scale to Batch Jobs], [
      - Reuse tuned single-file settings in multi-file processing.
      - Process many ASC files or CSV+image datasets.
      - Optional multiprocessing and per-trial output files.
      - Download consolidated ZIP result bundles.
    ])
  ],
)

#v(10pt)

#grid(
  columns: (3fr, 2fr),
  gutter: 12pt,
  [
    #card(rgb("#dde9ff"), [Recommended Operating Pattern], [
      1. Tune on representative single trials.
      2. Compare multiple correction algorithms.
      3. Lock parameters and save config JSON.
      4. Launch batch processing.
      5. Audit outputs at subject, trial, word, and sentence level.
      6. Use post-batch trial drill-down for quality assurance.
    ])
  ],
  [
    #card(rgb("#ffe8e8"), [What Makes GazeGenie Special], [
      - One interface spanning parsing → correction → analysis.
      - OCR-driven stimulus extraction for image-only workflows.
      - Cached OCR results for faster repeated runs.
      - Batch CSV+image support beyond classic ASC pipelines.
      - Eyekit integration with JSON export for reproducible alignment.
      - Interactive post-batch trial inspection to verify data quality.
    ])
  ],
)

#v(10pt)

#align(center)[
  #text(16pt, weight: "bold")[Landscape DIN A1 poster template for conference presentation]
]
