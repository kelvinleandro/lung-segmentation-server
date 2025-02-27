.PHONY: help install run-dev update-requirements clear code-formatting test

test: 
	pytest ./tests --rootdir=.

help:
	@echo "Local examples:"
	@echo "    make run-dev       		# Starts a uvicorn development server locally."
	@echo "    make install       		# Install dependencies from requirements.txt."
	@echo "    make clear         		# Removes cache directories."
	@echo "    make code-formatting 	# Formats the code using isort and black."

install:
	pip install --upgrade pip && \
	pip install --force-reinstall --upgrade --trusted-host pypi.python.org -r requirements.txt

run-dev:
	uvicorn main:app --timeout-keep-alive 60 --port 8080 --reload

clear:
	find . -type d -name '__pycache__' -exec rm -rf {} + && \
	find . -type d -name 'pytest_cache' -exec rm -rf {} + && \
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + && \
	find . -type d -name '.ruff_cache' -exec rm -rf {} + && \
	find . -type f -name '*.Zone.Identifier' -exec rm -f {} +

code-formatting:
	pip install --upgrade --quiet --force-reinstall isort ruff black 'black[jupyter]' && \
	isort . && \
	black . && \
	ruff format
