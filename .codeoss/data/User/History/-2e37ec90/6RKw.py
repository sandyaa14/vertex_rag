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

# ✅ Manually retrieve documents
try:
    retrieved_docs = rag.list_files(CORPUS_ID)  # ✅ Correct retrieval method

    if not retrieved_docs:
        print("❌ No documents retrieved. Check if the PDF was processed correctly.")
        exit()

    # Extract text from retrieved documents (simulate retrieval)
    retrieved_texts = [f"Document: {doc.name}" for doc in retrieved_docs]

except Exception as e:
    print(f"❌ Error retrieving documents: {e}")
    exit()

# ✅ Initialize GenerativeModel (NO retrieval argument)
model = GenerativeModel(model_name="gemini-1.0-pro-002")

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# ✅ Generate response using retrieved text as context
try:
    response = model.generate_content(
        [Part.from_text(query)] + [Part.from_text(text) for text in retrieved_texts],
        generation_config={"temperature": 0.2},  # Adjust as needed
    )

    # Debugging: Print the response
    print(f"📄 Full Response Object: {response}")

    # ✅ Process and display retrieved text
    if response and response.candidates:
        print("✅ Generated Response:")
        print(response.candidates[0].content.parts[0].text)

except Exception as e:
    print(f"❌ Error during query execution: {e}")
