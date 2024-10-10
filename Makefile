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
	black src/
	pylint src/