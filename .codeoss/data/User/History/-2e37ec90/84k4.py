import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, GroundingSource
from vertexai.preview import rag

# Project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get the first available corpus dynamically
try:
    corpora = list(rag.list_corpora())
    if not corpora:
        print("❌ No RAG Corpora found. Please create one first.")
        exit()

    CORPUS_ID = corpora[0].name
    print(f"✅ Using RAG Corpus: {CORPUS_ID}")

except Exception as e:
    print(f"❌ Error fetching corpora: {e}")
    exit()

# ✅ Correct way to configure retrieval using GroundingSource
retrieval_source = GroundingSource(vertex_rag=rag.VertexRagStore(rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)]))

# ✅ Initialize GenerativeModel with retrieval enabled
model = GenerativeModel(
    model_name="gemini-1.0-pro-002",
    grounding=retrieval_source  # ✅ Proper way to set RAG retrieval
)

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Generate response
response = model.generate_content(
    [Part.from_text(query)],
    generation_config={"temperature": 0.2},  # Adjust as needed
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
