# Setup

Cerberus requires Python 3.11 or newer. Run the commands below from the project root. The package and its dependencies are defined in [pyproject.toml](pyproject.toml).

## Create an environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Check the installation with `python -m examples.basic_rag`. It runs locally with the default policy and prints a response with an email address redacted.

## Optional components

Install the extras you need from the project root. Combine them in one command, for example `python -m pip install -e ".[dev,dashboard]"`.

| Extra | Purpose |
|---|---|
| `dashboard` | Streamlit dashboard |
| `dev` | Tests, lint, type checking, and package builds |
| `ml` | Local transformer-based input classifier |
| `ner` | spaCy support for detecting person names |
| `benchmark` | Public benchmark dataset support |
| `examples` | LangChain and LangGraph examples |

After installing `ner`, install the English model with `python -m spacy download en_core_web_sm`. Optional model components may download model files on first use. See the [input guide](docs/input.md) and [output guide](docs/output.md) for configuration.

## Start the gateway

In PowerShell, generate an API key for the current shell and start the server:

```powershell
$env:CERBERUS_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
python -m uvicorn cerberus.gateway.app:create_app --factory --host 127.0.0.1 --port 8000
```

On macOS or Linux, set the key using `export CERBERUS_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"` before running the same Uvicorn command. Open `http://127.0.0.1:8000/docs` for the API interface and send the key in the `X-API-Key` header.

For persistent local settings, copy [.env.example](.env.example) to `.env`, set `CERBERUS_API_KEY`, and add `--env-file .env` to the Uvicorn command. The default chat provider expects Ollama and the `qwen2.5:0.5b` model. See the [gateway instructions](README.md#run-the-gateway) for other providers.

## Run the dashboard demo

Install the dashboard extra and generate synthetic activity:

```sh
python -m pip install -e ".[dashboard]"
python -m examples.demo
```

Set `CERBERUS_DATABASE_URL` to `sqlite:///reports/demo.sqlite3`, using `$env:CERBERUS_DATABASE_URL = "sqlite:///reports/demo.sqlite3"` in PowerShell or `export CERBERUS_DATABASE_URL=sqlite:///reports/demo.sqlite3` on macOS and Linux. Run `cerberus-dashboard` and open `http://127.0.0.1:8501`.

## Verify the project

After installing the `dev` extra:

```sh
python -m ruff check .
python -m mypy
python -m pytest -q --cov=cerberus --cov=dashboard --cov-fail-under=80
python -m build
```

The [README](README.md) explains the architecture, policies, benchmark results, and Docker setup. The [benchmark guide](benchmarks/README.md) explains how to reproduce the evaluation.
