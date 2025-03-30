import vertexai
from vertexai.preview.generative_models import GenerativeModel, Retrieval, Part
from vertexai.preview import rag

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get the first available corpus dynamically
try:
    corpora = rag.list_corpora()
    if not corpora:
        raise ValueError("❌ No RAG corpus found. Please create one first.")
    corpus_name = corpora[0].name
    print(f"✅ Using RAG Corpus: {corpus_name}")
except Exception as e:
    print(f"❌ Error fetching corpora: {e}")
    exit()

# Ensure the corpus has files before querying
files = rag.list_files(corpus_name)
if not files:
    raise ValueError("❌ No files found in corpus! Import PDFs first.")

# Create a GenerativeModel with retrieval
try:
    model = GenerativeModel(
        model_name="gemini-1.0-pro-002",
        retrieval=Retrieval(
            vector_db=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_name)]
            )
        ),
    )
except Exception as e:
    print(f"❌ Error initializing Generative Model: {e}")
    exit()

# Define a sample query
query = "What is the document about?"
print(f"🔎 Querying RAG with: {query}")

# Retrieve relevant information
response = model.generate_content([Part.from_text(query)])

# Debugging: Print the response
print(f"📄 Full Response Object: {response}")

# Process and display retrieved text
if response and response.candidates:
    print("✅ Retrieved Documents:")
    for doc in response.candidates:
        print(f"- {doc.content.parts[0].text[:500]}...")  # Print first 500 characters for preview
else:
    print("❌ No documents retrieved. Check if the PDF was processed correctly.")
