# RAG_Production_App

## Inngest Server

<img src="RAG_APP/1.png" alt="App Screenshot" width="500" />


## Inngest Data

#### 1. Invoke the ingest function

<img src="RAG_APP/2.png" alt="App Screenshot" width="500" />


#### 2. Run observability

<img src="RAG_APP/3.png" alt="App Screenshot" width="500" />


## Query The Vector Database

#### 1. Invoke the query function

<img src="RAG_APP/4.png" alt="App Screenshot" width="500" />


#### 2. Run observability

<img src="RAG_APP/5.png" alt="App Screenshot" width="500" />


## Streamlit FrontEnd

<img src="RAG_APP/6.png" alt="App Screenshot" width="500" />


# Prerequisities

### Install UV

```
winget install --id=astral-sh.uv -e
```

### Install NodeJS

```
winget install OpenJS.NodeJS.LT
```

# Run Qdrant DB

```
docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

# Run APP Server

```
uv run uvicorn main:app
```

# Run Streamlit Frontend
```
uv run streamlit run .\streamlite_app.py
```

# Run Inngest Server

```
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

# Check Qdrant Collections

```
curl http://localhost:6333/collections/docs
```