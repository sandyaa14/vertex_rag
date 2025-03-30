from vertexai import rag
import vertexai

# Initialize Vertex AI (run this only once per session)
PROJECT_ID = "my-rag-project-455210"  # Replace with your project ID
vertexai.init(project=PROJECT_ID, location="us-central1")

# Configure the embedding model (text-embedding-005)
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# Create the RAG Corpus
display_name = "my_rag_corpus"
backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    backend_config=backend_config,
)

print("✅ RAG Corpus Created Successfully:", rag_corpus.name)
