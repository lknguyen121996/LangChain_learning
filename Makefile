# Setup python environment using pipfile
.PHONY: setup
setup:
	pipenv install --dev
	pipenv shell
# Run tests using pytest