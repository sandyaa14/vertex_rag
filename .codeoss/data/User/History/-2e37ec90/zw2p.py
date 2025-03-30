import vertexai
from vertexai.preview import rag

# TODO: Update project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# ✅ Use the correct RAG Corpus ID from check_corpora.py output
CORPUS_ID = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"  # Replace if needed

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# ✅ Fix: Use Retrieval with source (Corpus ID)
retrieval = rag.Retrieval(source=CORPUS_ID)

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve relevant information
response = retrieval.retrieve(query=query)

# ✅ Debugging: Print the response
print(f"📄 Full Response Object: {response}")

# ✅ Process and display retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for doc in response.candidates:
        print(f"- {doc.chunk.text[:500]}...")  # Print first 500 characters for preview
else:
    print("❌ No documents retrieved. Check if the PDF was processed correctly.")
