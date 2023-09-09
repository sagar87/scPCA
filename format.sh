poetry run black .
poetry run isort . --profile black
poetry run flake8 .
poetry run coverage run -m --source=scpca pytest tests
poetry run coverage report
