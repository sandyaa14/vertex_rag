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

# ✅ Initialize GenerativeModel
model = GenerativeModel(model_name="gemini-1.0-pro-002")

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# ✅ Retrieve relevant documents from the corpus
try:
    retrieval_source = rag.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)]
    )
    
    retrieved_docs = retrieval_source.retrieve([query])  # ✅ Corrected retrieval method

    if not retrieved_docs:
        print("❌ No documents retrieved. Check if the PDF was processed correctly.")
        exit()

    # Extract text from retrieved documents
    retrieved_texts = [doc.content for doc in retrieved_docs]

    # ✅ Generate response using retrieved text as context
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
