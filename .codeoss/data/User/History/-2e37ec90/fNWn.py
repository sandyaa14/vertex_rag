import vertexai
from vertexai.preview import rag

# Update your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
CORPUS_ID = "6917529027641081856"  # Update with the correct one, just the ID

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create RAG retrieval object with the correct source (Corpus ID)
retrieval = rag.Retrieval(corpus_id=CORPUS_ID)

# Define a test query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve documents based on the query
try:
    response = retrieval.retrieve(query=query)
except Exception as e:
    print(f"❌ An error occurred during retrieval: {e}")
    response = None

# Debugging: Print the full response
print(f"📄 Full Response Object: {response}")

# Extract and print retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for doc in response.candidates:
        print(f"- {doc.chunk.text[:500]}...")  # Print first 500 characters
else:
    print("❌ No documents retrieved. Check if the PDF was processed correctly.")
