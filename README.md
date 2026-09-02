 # LoanAssist

LoanAssist is a preliminary loan-eligibility assistant using policy retrieval,
a local Qwen model, LiteLLM, API guardrails, prompt versioning, and Streamlit.

This project provides preliminary assessment only; it does not replace normal
loan verification or approval.

## Architecture

Docker Compose starts the backend services:

| Service | URL | Purpose |
| --- | --- | --- |
| API | http://localhost:8001 | FastAPI `/ask`, `/health`, `/metrics` |
| LiteLLM | http://localhost:4000 | OpenAI-compatible provider proxy |
| Local model | http://localhost:8090 | Qwen model server |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Metrics dashboard (`admin` / `admin`) |

The Streamlit UI runs on the host at http://localhost:8501. ChromaDB is
embedded in the API container and persists under `rag/chroma/`.

## Prerequisites

- Git
- Docker Engine and Docker Compose v2
- Python 3.11 or newer
- About 1 GB of disk space for the local model

Check the installed versions:

```bash
docker --version
docker compose version
python --version
```

## Install docker and docker compose if not installed
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
```

## First-time setup

Run commands from the repository root:

```bash
cd /path/to/loan_eligibility_assistant
```

Create `.env` with:

```dotenv
API_KEY=local-dev-key
PROMPT_VERSION=v1
OPENROUTER_API_KEY=
```

Export your host user and group IDs before building the images so files written
to mounted volumes remain owned by you:

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
```

Compose defaults both values to `1000` when they are not exported.

Create the writable bind-mount directories before starting Docker. This is
important when a fresh checkout does not contain the empty `logs/` directory:

```bash
mkdir -p logs rag/chroma
chown "$HOST_UID:$HOST_GID" logs rag/chroma
```

If your checkout already contains files in these directories, use the same
command to give the mapped container user write access.

`PROMPT_VERSION` can be `v1` or `v2`; both are defined in
`prompts/registry.yaml`. The OpenRouter key is optional for the local model,
but is needed for the configured cloud fallback.

Create a host environment for the UI and RAG ingestion:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r ui/requirements.txt
pip install -r api/requirements.txt
```

The model server dependencies are installed inside its Docker image.

Build the RAG index after checkout and whenever a policy file changes:

```bash
python rag/ingest.py
```

This recreates the `eligibility` ChromaDB collection. Its first run downloads
an approximately 80 MB embedding model.

## Run the application

Start the backend stack:

```bash
docker compose up -d --build
docker compose ps
```

The local model may take several minutes to become healthy on first startup.
Check readiness with:

```bash
curl http://localhost:8090/health
curl http://localhost:8001/health
```

The API health response should report `"status":"ok"` and a positive
`rag_chunks` count. In a second terminal, start the UI:

```bash
source .venv/bin/activate
streamlit run ui/chat_app.py
```

Open http://localhost:8501. The UI defaults to `http://localhost:8001`.

## Useful commands

```bash
# Follow backend logs
docker compose logs -f api

# Stop services
docker compose down

# Stop services and remove Compose-managed volumes
docker compose down -v

# Rebuild the index after policy edits, then reload the API
python rag/ingest.py
docker compose restart api
```

## API example

```bash
curl -X POST http://localhost:8001/ask \
	-H 'Content-Type: application/json' \
	-H 'X-API-Key: local-dev-key' \
	-H 'Idempotency-Key: example-request-1' \
	-d '{"question":"What are the eligibility requirements?"}'
```

Metrics are available at http://localhost:8001/metrics. Grafana uses
`admin` / `admin` for local development.

## Troubleshooting

- **API unavailable:** inspect `docker compose ps` and
	`docker compose logs api litellm model`.
- **Model still starting:** wait for its health check; weights download on the
	first startup.
- **RAG errors or zero chunks:** run `python rag/ingest.py`, then
	`docker compose restart api`.
- **UI cannot connect:** confirm the sidebar API URL is
	`http://localhost:8001` and that the API health check succeeds.
- **Cloud fallback errors:** set a valid `OPENROUTER_API_KEY` in `.env` and
	run `docker compose up -d` again.

## Project layout

```text
api/             FastAPI application and guardrails
model_server/    OpenAI-compatible local Qwen server
litellm/         Provider proxy configuration
prompts/         Versioned prompt registry and loader
rag/             Eligibility policies and ChromaDB index
ui/              Streamlit chat application
prometheus/      Metrics scrape configuration
grafana/         Provisioned dashboards and data sources
scripts/         Canary, evaluation-gate, and rollback helpers
```