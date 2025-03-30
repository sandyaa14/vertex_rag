import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio
import time

# Step 1: Set up your Google Cloud Project ID
PROJECT_ID = "my-rag-project-455210"  # Update with your actual project ID
LOCATION = "us-central1"

# Step 2: Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Step 3: Check and Create RAG Corpus
corpus_display_name = "my_rag_corpus"
corpora = rag.list_corpora()

# Find existing corpus or create a new one
existing_corpus = next((c for c in corpora if c.display_name == corpus_display_name), None)

if existing_corpus:
    print(f"Using existing corpus: {corpus_display_name}")
    corpus = existing_corpus
else:
    print(f"Creating new corpus: {corpus_display_name}")
    corpus = rag.create_corpus(display_name=corpus_display_name)

# Step 4: Import Files from Google Cloud Storage
async def import_files():
    paths = ["gs://rag-bucket-sandyaakevin-12345/ch1.pdf"]  # Update with your GCS bucket path
    print("Importing files into RAG Corpus...")
    
    # Import files asynchronously
    response = await rag.import_files_async(corpus.name, paths)

    # Wait for completion
    result = await response.result()
    
    print("Files imported successfully:", result)
    return result

# Run the import_files function asynchronously
asyncio.run(import_files())

# Step 5: Wait until the file is indexed
print("Checking file status...")
while True:
    files = rag.list_files(corpus.name)
    if not files:
        print("No files found in corpus. Retrying...")
        time.sleep(5)
        continue

    file_statuses = [f.file_status.state for f in files]
    print("Current file statuses:", file_statuses)

    if all(status == "ACTIVE" for status in file_statuses):
        print("All files are indexed and ready for retrieval.")
        break
    else:
        print("Waiting for files to be indexed...")
        time.sleep(10)  # Wait before rechecking

# Step 6: Retrieve Context from RAG (FIXED)
print("Running a RAG retrieval query...")

retrieval = rag.Retrieval(
    source=rag.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    ),
)

# Running the query (NOW WITH retrieval_config)
try:
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
        text="Summarize the chapter",
        retrieval_config=rag.RagRetrievalConfig()
    )
    
    # Extract text response safely
    if response:
        print("RAG Response:", response)
    else:
        print("No response from RAG.")
except Exception as e:
    print("Error during retrieval:", e)

# Step 7: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

# Generate response
gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")

# Extract generated text safely
if gen_response and hasattr(gen_response, "text"):
    print("Generated Response:", gen_response.text)
else:
    print("No generated response received.")
