## Check installed docker version
```bash
docker --version
docker compose version
```

## Install docker and docker compose if not installed
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
```

## Copy example env file to create env file
```bash
cp .env.example .env
```

## Update API_KEY, PROMPT_VERSION, and OPENROUTER_API_KEY in .env file
```bash
API_KEY=*type any random key* # API_KEY=XYZ
PROMPT_VERSION=v1/v2
OPENROUTER_API_KEY=
```

## Create a virtualenv and install dependencies
```bash
python -m virtualenv .venv
source .venv/bin/activate      # if not already active
```

## Install dependencies mentioned in requirements file
```bash
pip install -r api/requirements.txt
pip install -r model_server/requirements.txt
pip install -r ui/requirements.txt
```

## Run the dockers
```bash
docker compose up -d --build
docker compose ps
```

## Run the chatbot ui
```bash
streamlit run ui/chat_app.py   # opens http://localhost:8501
```