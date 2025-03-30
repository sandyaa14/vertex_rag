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

# ✅ Retrieve documents and fetch actual content
retrieved_texts = []
try:
    retrieved_docs = list(rag.list_files(CORPUS_ID))  # Get list of files

    if not retrieved_docs:
        print("❌ No documents retrieved. Check if the PDF was processed correctly.")
        exit()

    for doc in retrieved_docs:
        print(f"📄 Retrieving document: {doc.name}")

        # ✅ Fetch document content (ensure proper text extraction)
        doc_content = doc.get_text() if hasattr(doc, "get_text") else None

        if doc_content:
            retrieved_texts.append(doc_content)
        else:
            print(f"⚠️ Warning: No text found in {doc.name}")

except Exception as e:
    print(f"❌ Error retrieving document contents: {e}")
    exit()

# ✅ Debug: Print retrieved text for verification
print("\n🔍 Retrieved Texts from RAG:")
for text in retrieved_texts:
    print(f"- {text[:500]}...\n")  # Show first 500 characters

# ✅ Initialize GenerativeModel
model = GenerativeModel(model_name="gemini-1.0-pro-002")

# Define a sample query
query = "What is the document about?"
print(f"\n🔎 Querying RAG with: {query}")

# ✅ Generate response using retrieved text as context
try:
    response = model.generate_content(
        [Part.from_text(query)] + [Part.from_text(text) for text in retrieved_texts],
        generation_config={"temperature": 0.2},  # Adjust as needed
    )

    # Debugging: Print the response
    print(f"\n📄 Full Response Object: {response}")

    # ✅ Process and display response
    if response and response.candidates:
        print("\n✅ Generated Response:")
        print(response.candidates[0].content.parts[0].text)

except Exception as e:
    print(f"❌ Error during query execution: {e}")
