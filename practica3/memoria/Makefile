all: memoria.pdf
%.pdf: %.md
	pandoc -F pandoc-citeproc -o $@ $< --pdf-engine=xelatex
