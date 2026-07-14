from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

# Check if the collection exists and get information about it
try:
    info = client.get_collection(collection_name="docs")
    print(f"Collection status: {info.status}")
    print(f"Points count: {info.points_count}")
    print(f"Vectors configuration: {info.config.params.vectors}")
except Exception as e:
    print(f"Error reading collection: {e}")