#!/bin/bash
# =============================================================================
# build.sh — Compile the Technical Report PDF
# =============================================================================
#
# PURPOSE:
#   This script assembles the full research article from individual Markdown
#   chapter files and compiles it into a publication-quality PDF using Pandoc
#   and the XeLaTeX typesetting engine.
#
# PREREQUISITES (Fedora / RPM-based Linux):
#   sudo dnf install pandoc texlive-scheme-basic texlive-xetex
#   Older Pandoc versions also need pandoc-citeproc for citation processing.
#
# HOW TO RUN:
#   chmod +x build.sh      # Make executable (only needed once)
#   ./build.sh             # Run the script
#
# OUTPUT:
#   build/final_technical_report.pdf
#
# HOW IT WORKS:
#   Pandoc reads multiple Markdown files, merges them in the order listed,
#   applies the custom LaTeX template for typography, resolves @citation
#   references via citeproc using the BibTeX database, and renders a PDF
#   via XeLaTeX (which supports modern fonts and Unicode).
# =============================================================================

# -e : Exit immediately if any command returns a non-zero exit code.
# This prevents silent failures (e.g., if pandoc is not installed).
set -e

# Check build prerequisites before creating any output.
for required_command in pandoc xelatex; do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "Error: required command '$required_command' is not installed." >&2
        echo "Install Pandoc and XeLaTeX; see README.md for setup instructions." >&2
        exit 1
    fi
done

# Pandoc added built-in citeproc support in version 2.11. Ubuntu 22.04 ships
# Pandoc 2.9, which requires the separate pandoc-citeproc filter.
if pandoc --help | grep -q -- '--citeproc'; then
    citation_options=(--citeproc)
elif command -v pandoc-citeproc >/dev/null 2>&1; then
    citation_options=(--filter=pandoc-citeproc)
else
    echo "Error: this Pandoc version needs the 'pandoc-citeproc' filter." >&2
    echo "Install it with: sudo apt install pandoc-citeproc" >&2
    exit 1
fi

# Create the build output directory if it doesn't already exist.
# -p flag means "no error if it already exists, create parent dirs too".
mkdir -p build

echo "Compiling Markdown sections into PDF..."

# -----------------------------------------------------------------------------
# PANDOC COMMAND BREAKDOWN
# -----------------------------------------------------------------------------
# pandoc                        → The document converter (Markdown → PDF)
#   metadata.yaml               → Document front-matter: title, author, date,
#                                  abstract, font size, page geometry, bibliography
#                                  paths, citation style, etc.
#   sections/*.md               → Glob pattern: feeds ALL .md files from the
#                                  sections/ directory, sorted alphabetically
#                                  (which equals numerical order: 01_, 02_, ...)
#   --template=template.tex     → Custom XeLaTeX template controlling the
#                                  visual layout: fonts, margins, header/footer,
#                                  section numbering style, figure captions, etc.
#   --pdf-engine=xelatex        → Use XeLaTeX as the PDF renderer. XeLaTeX
#                                  (vs. pdflatex) supports UTF-8 natively and
#                                  modern system fonts via fontspec.
#   --top-level-division=chapter → Maps top-level Markdown H1 headings (#)
#                                  to LaTeX \chapter{} commands so the PDF
#                                  has proper chapter breaks and numbering.
#   --toc                       → Generate a Table of Contents automatically
#                                  based on the section headings in the .md files.
#   --number-sections           → Auto-number all headings: 1, 1.1, 1.1.2 etc.
#   citation processor           → Uses built-in citeproc when supported; older
#                                  Pandoc versions use the pandoc-citeproc filter.
#   -o build/final_technical_report.pdf → Output file path.
# -----------------------------------------------------------------------------

pandoc metadata.yaml \
    sections/*.md \
    --template=template.tex \
    --pdf-engine=xelatex \
    --top-level-division=chapter \
    --toc \
    --number-sections \
    "${citation_options[@]}" \
    -o build/final_technical_report.pdf

echo "Build complete. Check build/final_technical_report.pdf"
