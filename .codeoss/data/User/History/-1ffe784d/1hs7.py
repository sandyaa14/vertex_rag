from vertexai import rag
import vertexai

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define RAG Corpus details
display_name = "my_rag_corpus"
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005@001"
    )
)
backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)

# Check if a corpus with the same name already exists
try:
    corpora = rag.list_corpora()
    existing_corpus = next((c for c in corpora if c.display_name == display_name), None)
except Exception as e:
    print(f" Error fetching corpora: {e}")
    corpora = []
    existing_corpus = None

if existing_corpus:
    print(f"RAG Corpus '{display_name}' already exists: {existing_corpus.name}")
    rag_corpus = existing_corpus
else:
    try:
        rag_corpus = rag.create_corpus(
            display_name=display_name,
            backend_config=backend_config,
        )
        print(f"RAG Corpus Created Successfully: {rag_corpus.name}")
    except Exception as e:
        print(f"Error creating RAG Corpus: {e}")
