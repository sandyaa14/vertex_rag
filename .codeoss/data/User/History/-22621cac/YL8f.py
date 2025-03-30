import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

# Step 1: Set up your Google Cloud Project ID
PROJECT_ID = "my-rag-project-455210"  # Update with your actual project ID
LOCATION = "us-central1"  # Keep this for Vertex AI

# Step 2: Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Step 3: Create RAG Corpus
display_name = "my_rag_corpus"

embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)
rag_corpus = rag.create_corpus(display_name=display_name, backend_config=backend_config)

# Step 4: List all RAG Corpora
print("Available RAG corpora:")
print(rag.list_corpora())

# Step 5: Import Files from Google Drive or Cloud Storage
corpus_name = rag_corpus.name  # Use the corpus created above
paths = ["gs://rag-bucket-sandyaakevin-12345/ch1.pdf"]  # Update with GCS bucket or Drive link

transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
)

rag.import_files(
    corpus_name,
    paths,
    transformation_config=transformation_config,
    max_embedding_requests_per_min=1000
)

# Step 6: Retrieve Context from RAG
rag_retrieval_config = rag.RagRetrievalConfig(top_k=3, filter=rag.Filter(vector_distance_threshold=0.5))

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
    text="What is RAG and why is it useful?",
    rag_retrieval_config=rag_retrieval_config,
)

print("RAG Response:", response)

# Step 7: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

# Generate content based on imported knowledge
gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")
print("Generated Response:", gen_response.text)
