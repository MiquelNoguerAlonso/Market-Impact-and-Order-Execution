# Market Microstructure Trilogy — Paper I

**The Mathematics of Market Impact and Order Execution**  
DOI: <https://doi.org/10.5281/zenodo.22736805>  
Fixed manuscript date: **15 September 2026**

## Compile

The main document is `main.tex`. With pdfLaTeX and `latexmk` installed, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

All 11 publication-resolution PNG figures and every section and table input
needed to compile the deposited PDF are included. The bibliography is embedded
in `main.tex`.

## Reproduce and verify

The numerical environment is pinned in `requirements.txt`. To regenerate the
figures, tables, result files, and numerical audit reports, run:

    python -m pip install -r requirements.txt
    python scripts/reproduce.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The reproduction command runs ten scripts and stops on any failed assertion.
To verify the hashes recorded for a delivered source tree before regenerating
it, run `python scripts/check_package.py`.

The `lean/` directory contains the pinned Lean 4.19.0/Mathlib v4.19.0
companion. Its exact scope and build commands are documented in
`lean/COMPILE.md`; the automated clean check is:

    python scripts/verify_lean.py --clean

The associated build, axiom, environment, and machine-readable audit evidence
is included under `audit/`.

## Package contents

- `main.tex`, `sections/`, and `tables/`: manuscript source and generated inputs.
- `figures/`: the 11 figures used by the manuscript.
- `scripts/`, `results/`, and `requirements.txt`: complete numerical reproduction.
- `lean/`: pinned formal companion sources and configuration.
- `audit/`: current verification evidence and the submission review.
- `Trilogy_Citations.bib`: definitive BibTeX records for the three papers.
- `MANIFEST_SHA256.txt`: SHA-256 inventory of the source archive.

The trilogy uses a fixed star citation architecture: Papers II and III cite
Paper I; Papers II and III do not cite one another.
