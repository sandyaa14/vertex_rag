import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
from google.cloud import storage
from docx import Document
import asyncio

# Step 1: Set up your Google Cloud Project ID
PROJECT_ID = "my-rag-project-455210"  # Update with your actual project ID
LOCATION = "us-central1"

# Step 2: Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Step 3: Check and Create RAG Corpus
corpus_display_name = "my_rag_corpus"
corpora = rag.list_corpora()
existing_corpus = next((c for c in corpora if c.display_name == corpus_display_name), None)

if existing_corpus:
    print(f"✅ Using existing corpus: {corpus_display_name}")
    corpus = existing_corpus
else:
    print(f"📌 Creating new corpus: {corpus_display_name}")
    corpus = rag.create_corpus(display_name=corpus_display_name)

# Step 4: Function to Extract Text from a .docx File
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Step 5: Import Files from Google Cloud Storage and Process Content
async def import_files():
    # Step 5.1: Define the path to your GCS bucket
    paths = ["gs://rag-bucket-sandyaakevin-12345/ch2.docx"]  # Update with your GCS bucket path
    print("📂 Importing files into RAG Corpus...")
    
    # Step 5.2: Download the file from GCS bucket
    def download_blob(bucket_name, source_blob_name, destination_file_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"File downloaded to {destination_file_name}")
    
    # Download the file locally
    download_blob('rag-bucket-sandyaakevin-12345', 'ch2.docx', 'ch2.docx')

    # Step 5.3: Extract text from the downloaded .docx file
    document_text = extract_text_from_docx('ch2.docx')
    print("📂 Extracted document text:", document_text[:500])  # Print first 500 characters for preview

    # Step 5.4: Add the extracted content to RAG corpus
    # In case you need to process it as a new document or update the corpus
    await rag.import_files_async(corpus.name, [document_text])
    print("✅ Files imported successfully.")
    print("📂 Files in RAG Corpus:", rag.list_files(corpus.name))

# Step 6: Run the import asynchronously
asyncio.run(import_files())

# Step 7: Retrieve Context from RAG
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

# Step 8: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

gen_response = rag_model.generate_content("Summarize the document")
print("Generated Response:", gen_response.text)
