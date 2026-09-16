# Market Microstructure Trilogy — submission review

Review date: **15 September 2026**

## Release decision

The three manuscripts are cleared for packaging as separate records. This
review did not upload files or alter any external record.

| Paper | DOI | Pages | PNG figures |
| --- | --- | ---: | ---: |
| I. *The Mathematics of Market Impact and Order Execution* | `10.5281/zenodo.22736805` | 117 | 11 |
| II. *Quoting Under a Misspecified Fill Law* | `10.5281/zenodo.22759528` | 39 | 5 |
| III. *From Forecast to Position* | `10.5281/zenodo.22759534` | 38 | 5 |

## Manuscript and build audit

- Every title page is fixed at 15 September 2026 and prints the correct DOI.
- PDF title, author, subject, keywords, and DOI metadata are populated; volatile
  engine dates and trailer identifiers are suppressed for reproducible builds.
- The trilogy retains its chosen star citation graph: Papers II and III cite
  Paper I; neither cites the other.
- All figures are included with widths expressed relative to `\linewidth`.
- The final pdfLaTeX/BibTeX builds have no unresolved references, undefined
  citations, duplicate labels, or overfull boxes. Paper I retains nine harmless
  underfull-box notices; Papers II and III have none.
- All fonts are embedded. All 194 pages were rasterized and visually reviewed,
  including the title pages, contents, equations, tables, bibliography, and all
  21 figures. No clipping, corruption, or missing content was found.

## Computational audit

- Paper I's ten-program `scripts/reproduce.py` workflow completed successfully,
  regenerating and checking its figures, tables, arrays, and numerical reports.
- The shared Paper II–III `verification/verify_theory.py` suite completed with
  `all_passed` status for all 41 check groups.
- Every PNG passed a decoder/CRC integrity check after regeneration. A truncated
  intermediate copy of Paper II Figure 4 was detected during the release build,
  regenerated in isolation, revalidated, and replaced before packaging.
- The supplied Lean record documents a clean Lean 4.19.0/Mathlib v4.19.0 build.
  All seven current Lean source/configuration hashes match that successful
  record, and the active source contains no admitted-proof or unsafe tokens.

## Source-package audit

- Paper I includes the complete LaTeX tree, 11 figures, generated tables,
  scripts, pinned Python requirements, result arrays, numerical audit records,
  Lean sources/configuration, and Lean build/axiom evidence.
- Papers II and III each include their complete LaTeX/BibTeX tree, five figures,
  and the complete rerunnable 41-group verification and figure-generation suite.
- Each source archive contains a README, the definitive trilogy BibTeX entries,
  and a SHA-256 manifest covering every other archived file.

The examples are synthetic and the guarantees remain conditional on the models,
information sets, and uncertainty bounds stated in the manuscripts.
