import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
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

# ✅ Initialize GenerativeModel (without retrieval)
model = GenerativeModel(model_name="gemini-1.0-pro-002")

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# ✅ Define the correct retrieval configuration
retrieval_source = rag.VertexRagStore(
    rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)]
)

# ✅ Generate response using `grounding_source`
response = model.generate_content(
    [Part.from_text(query)],
    generation_config={"temperature": 0.2},  # Adjust as needed
    grounding_source=retrieval_source,  # ✅ Corrected way to use RAG retrieval
)

# Debugging: Print the response
print(f"📄 Full Response Object: {response}")

# ✅ Process and display retrieved text correctly
if response and response.candidates:
    retrieved_docs = response.candidates[0].content.parts
    if retrieved_docs:
        print("✅ Retrieved Documents:")
        for doc in retrieved_docs:
            print(f"- {doc.text[:500]}...")  # Print first 500 characters for preview
    else:
        print("❌ No documents retrieved. Check if the PDF was processed correctly.")
else:
    print("❌ No response received. Check if the query and RAG setup are correct.")
