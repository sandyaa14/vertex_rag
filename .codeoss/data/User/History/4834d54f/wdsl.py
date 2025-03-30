from vertexai import rag
import vertexai

# Set up your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Choose the first corpus from the list
corpus_name = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"

# List files inside the corpus
files = rag.list_files(corpus_name)

if files:
    print("✅ Files in RAG Corpus:")
    for file in files:
        print(f"- {file.name}")
else:
    print("❌ No files found. Did you import the PDF?")
