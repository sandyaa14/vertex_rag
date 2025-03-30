import vertexai
from vertexai import rag

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get the first available corpus dynamically
try:
    corpora = list(rag.list_corpora())  # Convert pager to a list
    if not corpora:
        print("No RAG Corpora found. Please create one first.")
        exit()

    corpus_name = corpora[0].name  # Now this works correctly
    print(f"Using RAG Corpus: {corpus_name}")

except Exception as e:
    print(f"Error fetching corpora: {e}")
    exit()

# List files inside the corpus
try:
    files = list(rag.list_files(corpus_name))  # Convert pager to a list
    if files:
        print("Files in RAG Corpus:")
        for file in files:
            print(f"- {file.name}")
    else:
        print("No files found. Did you import the PDF?")
except Exception as e:
    print(f"Error listing files: {e}")
