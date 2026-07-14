# RAG_Production_App

# Prerequisities

### Install UV

``` winget install --id=astral-sh.uv -e ```

### Install NodeJS

``` winget install OpenJS.NodeJS.LT ```

# Run APP Server

``` uv run uvicorn main:ap ```

# Run Inngest Server

``` npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery ```

# Run Qdrant DB

``` docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant ```

# Check Qdrant Collections

``` curl http://localhost:6333/collections/docs ```