import vertexai
from vertexai.generative_models import GenerativeModel, Retrieval, Part
from vertexai.preview import rag

# Project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Corpus ID
CORPUS_ID = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Create a GenerativeModel with retrieval
model = GenerativeModel(
    model_name="gemini-1.0-pro-002",
    retrieval=Retrieval(
        vector_db=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)]
        )
    ),
)

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve relevant information
response = model.generate_content(
    [Part.from_text(query)],
)

# Debugging: Print the response
print(f"📄 Full Response Object: {response}")

# Process and display retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for doc in response.candidates:
        print(f"- {doc.content.parts[0].text[:500]}...")  # Print first 500 characters for preview
else:
    print("❌ No documents retrieved. Check if the PDF was processed correctly.")
