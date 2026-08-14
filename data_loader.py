from llama_index.readers.file import PDFReader, docs
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from fastembed import TextEmbedding
from qdrant_client import models


load_dotenv()



# embedding_model = TextEmbedding(
#     model_name="BAAI/bge-m3"
# )

embedding_model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # embeddings = list(embedding_model.embed(texts))
    embeddings = [
        {
        "dense": list(embedding_model.embed([text]))[0].tolist(),
        "sparse": models.Document(text=text, model="Qdrant/bm25")
        }
        for text in texts
    ]
    return embeddings