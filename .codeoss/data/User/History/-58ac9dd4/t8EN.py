import vertexai
from vertexai import rag

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# List available RAG Corpora
try:
    corpora = rag.list_corpora()
    if corpora:
        print("✅ Available RAG Corpora:")
        for corpus in corpora:
            print(f"- Name: {corpus.name}")
    else:
        print("❌ No RAG Corpora found. Did you create one?")
except Exception as e:
    print(f"⚠️ Error listing corpora: {e}")
