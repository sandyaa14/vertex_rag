from vertexai import rag
import vertexai

# Initialize Vertex AI (run this only once per session)
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure the embedding model (text-embedding-005)
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# Create the RAG Corpus
display_name = "my_rag_corpus"
backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)

# Check if a corpus with the same name already exists
corpora = rag.list_corpora()
existing_corpus = None
for corpus in corpora:
    if corpus.display_name == display_name:
        existing_corpus = corpus
        break

if existing_corpus:
    print(f"✅ RAG Corpus '{display_name}' already exists: {existing_corpus.name}")
    rag_corpus = existing_corpus
else:
    rag_corpus = rag.create_corpus(
        display_name=display_name,
        backend_config=backend_config,
    )
    print("✅ RAG Corpus Created Successfully:", rag_corpus.name)
