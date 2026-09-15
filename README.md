# The Mathematics of Market Impact and Order Execution

**Miquel Noguer Alonso**  
Artificial Intelligence Finance Institute (AIFI)

A theoretical and computational treatment of impact kernels, feasible execution policies, venue allocation, hidden liquidity, and execution certificates.

- DOI: [10.5281/zenodo.22736805](https://doi.org/10.5281/zenodo.22736805)
- Overleaf: [editable project](https://www.overleaf.com/project/6aa88ac6155ad0f8efdcc21b)
- Manuscript: [`paper.pdf`](paper.pdf)
- LaTeX: [`paper.tex`](paper.tex)

## Repository contents

This private repository is part 1 of the *Market Microstructure Trilogy*. It contains the reviewed manuscript, its LaTeX source, and the corresponding source and verification archive. The manuscript uses author-year citations and includes a table of contents.

## Build

The manuscript was built with pdfLaTeX. For Papers II and III, run BibTeX between LaTeX passes.

```bash
latexmk -pdf paper.tex
```

## Verification status

The released PDF was reproduced from the included source on 15 September 2026. The Overleaf build completed with zero errors and zero warnings. Numerical and symbolic checks are contained in the accompanying verification archive.

## Scope

The guarantees in the paper are conditional on the declared models, information sets, and uncertainty bounds. Synthetic calculations verify the stated identities and certificates; they do not claim live-market profitability.
