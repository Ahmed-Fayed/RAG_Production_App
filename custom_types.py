import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGChunkAndSRCDB(pydantic.BaseModel):
    chunks_list: list[list[str]]
    sources_ids: list[str]

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGUpsertDBResult(pydantic.BaseModel):
    ingested_files: int
    ingested_chunks: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
