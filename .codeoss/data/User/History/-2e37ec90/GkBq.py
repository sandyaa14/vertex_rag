import vertexai
from vertexai.preview import rag

# Update your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
CORPUS_ID = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"  # Full corpus name

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create RAG retrieval object with the correct source (full corpus name)
try:
    retrieval = rag.Retrieval(source=CORPUS_ID)
except Exception as e:
    print(f"❌ Error initializing Retrieval object: {e}")
    exit()

# Define a test query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve documents based on the query
try:
    response = retrieval.retrieve(query=query)
except Exception as e:
    print(f"❌ An error occurred during retrieval: {e}")
    print(f"   Possible reasons: \n"
          f"   1. The corpus might be empty. Check using `list_files.py`.\n"
          f"   2. There might be an issue with the Vertex AI API. Check the status.\n"
          f"   3. The query might be too broad or not related to the document content.\n"
          f"   4. The corpus ID might be incorrect. Verify it using `check_corpora.py`.\n"
          f"   5. There might be a network issue.")
    response = None

# Debugging: Print the full response
print(f"📄 Full Response Object: {response}")

# Extract and print retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for i, doc in enumerate(response.candidates):
        print(f"  Document {i+1}:")
        print(f"  - Text: {doc.chunk.text[:500]}...")  # Print first 500 characters
        print(f"  - File Name: {doc.chunk.file_name}")
        print(f"  - Chunk ID: {doc.chunk.chunk_id}")
        print(f"  - Confidence Score: {doc.confidence_score}")
        print("-" * 20)
else:
    print("❌ No documents retrieved. Check the following:")
    print("   1. Verify that the PDF was imported and processed correctly using `list_files.py`.")
    print("   2. Ensure that the corpus ID is correct using `check_corpora.py`.")
    print("   3. Check if the query is relevant to the content of the imported PDF.")
