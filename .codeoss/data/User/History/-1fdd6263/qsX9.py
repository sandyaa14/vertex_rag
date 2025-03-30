import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio  # To handle async function calls

# Step 1: Set up your Google Cloud Project ID
PROJECT_ID = "my-rag-project-455210"  # Update with your actual project ID
LOCATION = "us-central1"  # Keep this for Vertex AI

# Step 2: Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Step 3: Create or Retrieve RAG Corpus
display_name = "my_rag_corpus"
existing_corpora = rag.list_corpora()
rag_corpus = None

for corpus in existing_corpora:
    if corpus.display_name == display_name:
        print(f"✅ Using existing corpus: {display_name}")
        rag_corpus = corpus
        break

if rag_corpus is None:
    print("🆕 Creating a new RAG corpus...")
    rag_corpus = rag.create_corpus(display_name=display_name)
    print(f"✅ New corpus created: {rag_corpus.name}")

# Step 4: Import Files into RAG Corpus (Using Async for proper handling)
async def import_files():
    paths = ["gs://rag-bucket-sandyaakevin-12345/ch1.pdf"]  # Update with your GCS bucket link
    print("📂 Importing files into RAG Corpus...")
    await rag.import_files_async(rag_corpus.name, paths)
    print("✅ Files imported successfully.")

asyncio.run(import_files())

# Step 5: Retrieve Context from RAG
print("🔍 Running a RAG retrieval query...")
retrieval_query = "What is RAG and why is it useful?"
response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
    text=retrieval_query
)
print("RAG Response:", response)

# Step 6: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(),  # Removed incorrect arguments
    )
)

rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")
print("Generated Response:", gen_response.text)
