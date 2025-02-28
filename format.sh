poetry run black .
poetry run isort . --profile black
poetry run flake8 scpca
poetry run flake8 tests
poetry run coverage run -m --source=scpca pytest tests
poetry run coverage report
