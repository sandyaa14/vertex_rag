import vertexai
from vertexai.preview import rag

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

# Define corpus name
corpus_name = "my_rag_corpus"

try:
    # Check if corpus exists
    corpus = rag.get_corpus(name=corpus_name)
    print(f"✅ Using existing corpus: {corpus_name}")
except Exception as e:
    print("❌ Corpus retrieval failed:", e)
    exit()

# Import files (if needed, else skip)
try:
    print("📂 Importing files into RAG Corpus...")
    import_response = rag.import_files_async(
        corpus=corpus,
        files=[rag.RagFile(uri="gs://your-bucket-name/sample.txt")]
    )
    import_response.result()  # Ensure completion
    print("✅ Files imported successfully.")
except Exception as e:
    print("❌ File import failed:", e)

# Configure retrieval
retrieval = rag.Retrieval(
    rag_store=rag.VertexRagStore(rag_corpus=corpus_name)
)

# Run RAG retrieval query
try:
    print("🔍 Running a RAG retrieval query...")
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text="Explain Retrieval-Augmented Generation in simple terms.",
        top_k=5
    )
    print("RAG Response:", response)
except Exception as e:
    print("❌ Retrieval query failed:", e)

# Troubleshooting tips
print("\n✅ If retrieval works but you still see warnings, try restarting Cloud Shell or checking your GCP quotas.")
