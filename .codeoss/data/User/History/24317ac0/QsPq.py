import asyncio
import fitz  # PyMuPDF for PDF text extraction
from google.cloud import storage
from vertexai.preview import rag
import vertexai

# 🔹 Set Google Cloud Project & Location
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# 🔹 Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# 🔹 Set up Google Cloud Storage (GCS) bucket
BUCKET_NAME = "rag-bucket-sandyaakevin-12345"
PDF_PATH = "/home/sandyaa2004/Try2/ch1.pdf"
TEXT_FILE_PATH = "/home/sandyaa2004/Try2/ch1.txt"
GCS_URI = f"gs://{BUCKET_NAME}/ch1.txt"

# 🔹 Initialize RAG corpus name
CORPUS_NAME = "my_rag_corpus"

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

# Step 2: Save extracted text to a local file
def save_text_to_file(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

# Step 3: Upload extracted text to Google Cloud Storage (GCS)
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"✅ Uploaded {source_file_name} to GCS as {destination_blob_name}")

# Step 4: Check or create the RAG corpus
def get_or_create_corpus(corpus_name):
    try:
        corpus = rag.get_corpus(name=corpus_name)
        print(f"✅ Using existing corpus: {corpus_name}")
    except Exception:
        corpus = rag.create_corpus(display_name=corpus_name)
        print(f"✅ Created new corpus: {corpus_name}")
    return corpus

# Step 5: Upload extracted text into RAG corpus
async def upload_text_to_rag(corpus):
    response = await rag.import_files(
        corpus=corpus,
        gcs_source_uri=GCS_URI  # ✅ Fixed: Use `gcs_source_uri`
    )
    print("✅ Extracted text uploaded successfully:", response)

# Step 6: Perform RAG retrieval query
async def run_rag_query(corpus):
    retriever = rag.get_retriever(corpus=corpus)  # ✅ Added retriever
    query = "What is the main topic of the document?"  
    response = await retriever.retrieval_query(query=query)  # ✅ Used retriever

    if hasattr(response, "contexts") and response.contexts:
        print("🔎 RAG Response Contexts:", response.contexts)
    else:
        print("❌ No relevant context found in the document.")
    
    print("💡 Generated Response:", response)

# Main Execution
if __name__ == "__main__":
    print(f"📢 Using GCP Project: {PROJECT_ID} at {LOCATION}")

    extracted_text = extract_text_from_pdf(PDF_PATH)
    print("📜 Extracted PDF Text (First 500 chars):", extracted_text[:500])

    save_text_to_file(extracted_text, TEXT_FILE_PATH)
    upload_to_gcs(BUCKET_NAME, TEXT_FILE_PATH, "ch1.txt")

    corpus = get_or_create_corpus(CORPUS_NAME)

    asyncio.run(upload_text_to_rag(corpus))
    asyncio.run(run_rag_query(corpus))
