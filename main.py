from itertools import count
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from openai import OpenAI
from dotenv import load_dotenv
import uuid
import os
import datetime
from custom_types import RAGChunkAndSrc, RAGChunkAndSRCDB, RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGUpsertDBResult
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage



load_dotenv()


adapter = ai.openai.Adapter(
    auth_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    model="openrouter/free"
)


inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)



@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
    limit=1,
    period=datetime.timedelta(minutes=1),
    key="event.data.source_id",
    burst=2,
    ),
    rate_limit=inngest.RateLimit(
    limit=1,
    period=datetime.timedelta(hours=4),
    key="event.data.source_id",
    )
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Ingest Dataset",
    trigger=inngest.TriggerEvent(event="rag/ingest_db"),
    throttle=inngest.Throttle(
        limit=1,
        period=datetime.timedelta(minutes=1),
        key="event.data.source_id",
        burst=2
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    )
)
async def rag_ingest_db(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSRCDB:
        db_path = ctx.event.data["db_path"]
        db_chunks = []
        db_sources = []
        for pdf_file in os.listdir(db_path):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(db_path, pdf_file)
                chunks = load_and_chunk_pdf(pdf_path)
                db_chunks.append(chunks)
                db_sources.append(pdf_file)
        return RAGChunkAndSRCDB(chunks_list=db_chunks, sources_ids=db_sources)
    
    def _upsert(chunks_list_and_srcs: RAGChunkAndSRCDB) -> RAGUpsertDBResult:
        chunks_list = chunks_list_and_srcs.chunks_list
        sources_ids = chunks_list_and_srcs.sources_ids
        ingested_files, ingested_chunks = 0, 0
        for chunks, source_id in zip(chunks_list, sources_ids):
            num_chunks = len(chunks)
            vecs = embed_texts(chunks)
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(num_chunks)]
            payloads = [{"source": source_id, "text": chunks[i]} for i in range(num_chunks)]
            QdrantStorage().upsert(ids, vecs, payloads)
            ingested_files += 1
            ingested_chunks += num_chunks
        return RAGUpsertDBResult(ingested_files=ingested_files, ingested_chunks=ingested_chunks)
    
    
    rag_and_src_db = await ctx.step.run("load-and-chunk-db", lambda: _load(ctx), output_type=RAGChunkAndSRCDB)
    ingested_files, ingested_chunks = await ctx.step.run("embed-and-upsert-db", lambda: _upsert(rag_and_src_db), output_type=RAGUpsertDBResult)
    
    return {"ingested_files": ingested_files, "ingested_chunks": ingested_chunks}




@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, topk: int = 5, search_type: str = "hybrid"):
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        
        if search_type == "dense":
            found = store.dense_search(query_vec, topk)
        elif search_type == "sparse":
            found = store.sparse_search(query_vec, topk)
        else:
            found = store.rrf_search(query_vec, topk)
        
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    
    question = ctx.event.data["question"]
    topk = int(ctx.event.data.get("topk", 5))
    search_type = ctx.event.data.get("search_type", "hybrid")

    found = await ctx.step.run("embed-and-search", lambda: _search(question, topk, search_type), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    
    res = await ctx.step.ai.infer(
        "LLM-Answer",
        adapter=adapter,
        body={
            # "model": "deepseek-chat",
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}



    




app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_ingest_db, rag_query_pdf_ai])
