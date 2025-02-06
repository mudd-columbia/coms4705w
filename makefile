get_edited_python_files = $(shell git status --porcelain | awk '!/[D?R]/ {print $$2}' | grep -E '.*\.(py)$$')

style:
	@echo $(get_edited_python_files)
	@edited_files="$(get_edited_python_files)"; \
	if [ -n "$$edited_files" ]; then \
		echo "Running black on edited python files..."; \
		black $$edited_files; \
		echo "Running ruff on edited python files..."; \
		ruff check $$edited_files; \
	else \
		echo "No python files have been edited."; \
	fi

style-all:
	@echo "Running black and isort on all Python files in hw0..."
	black hw0
	isort hw0

# ex.  make test hw=hw1
test:
	@echo "Running tests..."
	@if [ -z "$(hw)" ]; then \
		echo "Please specify a module to test with 'hw=<module_name>'"; \
		exit 1; \
	fi
	pytest $(hw)/tests.py || echo "Test failed."
