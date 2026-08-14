from qdrant_client import QdrantClient
from qdrant_client import models


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=768, force_recreate=False):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Delete the collection if force_recreate is set to True
        if force_recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)  # IDF: Inverse Document Frequency weighting for BM25 sparse vectors
                }
            )
    
    def upsert(self, ids, vectors, payloads):
        points = [models.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points)
    
    def search(self, query_vector, topk=10):
        results = self.client.query_points(
            collection_name=self.collection, 
            query=query_vector, 
            with_payload=True, 
            limit=topk)
        
        contexts = []
        sources =set()

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
    
    def dense_search(self, query_vector, topk=10):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector["dense"],
            using="dense",
            with_payload=True,
            limit=topk
        )
        
        contexts = []
        sources = set()
        
        for point in results.points:
            payload = getattr(point, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
    
    def sparse_search(self, query_vector, topk=10):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector["sparse"],
            using="sparse",
            with_payload=True,
            limit=topk
        )
        
        contexts = []
        sources = set()
        
        for point in results.points:
            payload = getattr(point, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
    
    def rrf_search(self, query_vector, topk=10):
        # Perfom hybrid search with Reciprocal Rank Fusion (RRF)
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector["dense"],
                    using="dense",
                    limit=topk
                ),
                models.Prefetch(
                    query=query_vector["sparse"],
                    using="sparse",
                    limit=topk
                )
            ],
            query = models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=topk
        )
        
        contexts = []
        sources = set()
        
        for point in results.points:
            payload = getattr(point, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}


