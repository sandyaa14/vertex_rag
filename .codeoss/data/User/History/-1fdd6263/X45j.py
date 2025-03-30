import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool

# ✅ Step 1: Set up Google Cloud Project
PROJECT_ID = "my-rag-project-455210"  # Update with actual project ID
LOCATION = "us-central1"  # Keep this for Vertex AI

# ✅ Step 2: Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ✅ Step 3: Create or Retrieve an Existing RAG Corpus
display_name = "my_rag_corpus"

# Check if corpus exists
existing_corpora = rag.list_corpora()
corpus = next((c for c in existing_corpora if c.display_name == display_name), None)

if not corpus:
    print(f"🚀 Creating a new RAG corpus: {display_name}...")

    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint="publishers/google/models/text-embedding-005"
    )

    backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)
    corpus = rag.create_corpus(display_name=display_name, backend_config=backend_config)
    print(f"✅ Corpus '{corpus.display_name}' created successfully.")
else:
    print(f"✅ Using existing corpus: {corpus.display_name}")

# ✅ Step 4: Import Files from Google Cloud Storage (GCS)
corpus_name = corpus.name
paths = ["gs://rag-bucket-sandyaakevin-12345/ch1.pdf"]  # Update with GCS file

# ✅ Fix: Use chunking config directly
chunking_config = rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)

print("📂 Importing files into RAG Corpus...")
import_response = rag.import_files_async(
    corpus_name=corpus_name,
    paths=paths,
    chunking_config=chunking_config,  # ✅ Pass chunking config directly
    max_embedding_requests_per_min=1000
)
import_response.result()  # Wait for completion
print("✅ Files imported successfully.")

# ✅ Step 5: Retrieve Context from RAG
retrieval_config = rag.RagRetrievalConfig(top_k=3, filter=rag.Filter(vector_distance_threshold=0.5))

print("🔍 Running a RAG retrieval query...")
response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
    text="What is RAG and why is it useful?",
    rag_retrieval_config=retrieval_config,
)

print("🤖 RAG Response:", response)

# ✅ Step 6: Use RAG-Enhanced Generation with Gemini Model
print("🚀 Setting up RAG-Enhanced Generation...")
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            rag_retrieval_config=retrieval_config,
        ),
    )
)

rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

# Generate content using RAG-enhanced retrieval
print("🧠 Generating content using RAG-enhanced AI...")
gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")
print("📝 Generated Response:", gen_response.text)
