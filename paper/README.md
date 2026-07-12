# Manuscript (LaTeX)

Living draft for publication. The intent is to keep the paper aligned with what
the repository actually implements (and to explicitly state scope limitations,
e.g., all tests at 25°C).

## Build

If you have `latexmk`:
```bash
cd paper
latexmk -pdf main.tex
```

Otherwise (classic BibTeX flow):
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Files
- `main.tex`: top-level manuscript
- `sections/`: modular sections (edit these most often)
- `references.bib`: BibTeX database
- `figures/`: place generated figures here (PNG/PDF)

