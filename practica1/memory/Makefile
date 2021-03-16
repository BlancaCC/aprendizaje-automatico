latex: main.tex bibliography.bib title_page.tex gradient_descent.tex
	pdflatex -shell-escape main
	bibtex main
	pdflatex -shell-escape main
	xdg-open main.pdf

clean:
	rm -r *~ _*  *.aux  *.bcf *.lof *.lot *.xml *.log *.out *.toc *.bbl *.blg *.dvi *-blx.bib
  

