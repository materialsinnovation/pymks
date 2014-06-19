# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

RST_DIR = rst
NOTEBOOK_DIR = ../notebooks
MARKDOWN_DIR = ..

NOTEBOOKS     = $(wildcard $(NOTEBOOK_DIR)/*.ipynb)
NOTEBOOKS_RST = $(patsubst ../notebooks/%.ipynb,$(RST_DIR)/%.rst,$(NOTEBOOKS))

MARKDOWN      = $(wildcard $(MARKDOWN_DIR)/*.md)
RST           = $(patsubst ../%.md,$(RST_DIR)/%.rst,$(MARKDOWN))

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: clean html

all: html

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(RST_DIR)
	rm -rf intro*

$(RST_DIR)/%.rst: $(MARKDOWN_DIR)/%.md
	mkdir -p $(RST_DIR)
	pandoc -f markdown -t rst $< -o $@

$(RST_DIR)/%.rst: $(NOTEBOOK_DIR)/%.ipynb
	mkdir -p $(RST_DIR)
	cd $(RST_DIR); ipython nbconvert --to rst ../$<

html: $(RST) $(NOTEBOOKS_RST) *.rst
	@cp rst/intro.rst ./
	@cp -r rst/intro_files ./
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

