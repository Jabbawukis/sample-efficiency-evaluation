# Top level makefile to build all

## help:                Show the help.
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep | sed 's/##/ -/g'

## run-workflow: Run the workflow pipeline locally for quick evaluation.
.PHONY: run-workflow
run-workflow:
	pip install .[dev]
	pytest tests/
	black src/
	pylint src/

## install: Install the package locally.
.PHONY: install
install:
	pip install .

## download-knowledge-base: Download the knowledge base for entity linking.
.PHONY: download-knowledge-base
download-knowledge-base:
	python -m spacy_entity_linker "download_knowledge_base"
	python -m spacy download en_core_web_md
	python -m spacy download en_core_web_trf
