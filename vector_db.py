from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=768, force_recreate=False):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Delete the collection if force_recreate is set to True
        if force_recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    
    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points)
    
    def search(self, query_vector, topk=10):
        results = self.client.query_points(
            collection_name=self.collection, 
            query=query_vector, 
            with_payload=True, 
            limit=topk)
        
        contexts = []
        sources =set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}


