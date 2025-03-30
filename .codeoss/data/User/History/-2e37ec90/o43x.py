import vertexai
from vertexai.preview import rag

# Update your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create RAG retrieval object
retrieval = rag.Retrieval()

# Define a test query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve documents based on the query
response = retrieval.retrieve(query=query)

# Debugging: Print the full response
print(f"📄 Full Response Object: {response}")

# Extract and print retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for doc in response.candidates:
        print(f"- {doc.chunk.text[:500]}...")  # Print first 500 characters
else:
    print("❌ No documents retrieved. Check if the PDF was processed correctly.")
