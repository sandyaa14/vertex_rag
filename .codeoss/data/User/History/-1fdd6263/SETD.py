import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio

# Setting up Google Cloud Project ID
PROJECT_ID = "my-rag-project-455210" 
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Check and Create RAG Corpus
corpus_display_name = "my_rag_corpus"
corpora = rag.list_corpora()
existing_corpus = next((c for c in corpora if c.display_name == corpus_display_name), None)

if existing_corpus:
    print(f"Using existing corpus: {corpus_display_name}")
    corpus = existing_corpus
else:
    print(f"Creating new corpus: {corpus_display_name}")
    corpus = rag.create_corpus(display_name=corpus_display_name)

# Import Files from Google Cloud Storage
async def import_files():
    paths = ["gs://rag-bucket-sandyaakevin-12345/ch2.docx"]  # Update with your GCS bucket path
    print("📂 Importing files into RAG Corpus...")
    await rag.import_files_async(corpus.name, paths)
    print("✅ Files imported successfully.")
    print("📂 Files in RAG Corpus:", rag.list_files(corpus.name))


asyncio.run(import_files())

# Step 5: Retrieve Context from RAG
print("🔍 Running a RAG retrieval query...")
retrieval = rag.Retrieval(
    source=rag.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    ),
)

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="Summarize the document"
)
print("RAG Response:", response)

# Step 6: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

gen_response = rag_model.generate_content("Summarize the document")
print("Generated Response:", gen_response.text)
