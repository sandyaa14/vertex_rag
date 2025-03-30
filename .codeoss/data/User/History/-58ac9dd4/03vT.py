from vertexai import rag
import vertexai

# Set up your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# List available RAG Corpora
corpora = rag.list_corpora()

if corpora:
    print("✅ Available RAG Corpora:")
    for corpus in corpora:
        print(f"- Name: {corpus.name}")
else:
    print("❌ No RAG Corpora found. Did you create one?")
