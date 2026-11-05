PY ?= python

.PHONY: demo test clean

demo:            ## full pipeline on synthetic data
	cd src && export MPLBACKEND=Agg && $(PY) make_synthetic.py && $(PY) build_panel.py && $(PY) analyze.py && $(PY) selftest.py

test:            ## planted-effect recovery + invariant checks
	$(PY) -m pytest -q tests

clean:           ## remove generated files from src/
	rm -f src/*.csv src/*.png
	rm -rf src/__pycache__
