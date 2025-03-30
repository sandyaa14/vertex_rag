import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import fitz  # PyMuPDF for PDF text extraction
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

# Step 4: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

pdf_path = "/home/sandyaa2004/Try2/ch1.pdf"  # Ensure this path is correct
extracted_text = extract_text_from_pdf(pdf_path)

print("📜 Extracted PDF Text (First 500 chars):", extracted_text[:500])

# Step 5: Upload Extracted Text as a File in RAG Corpus
async def upload_text_to_rag():
    """Upload extracted text to RAG corpus as a file."""
    file_path = "gs://rag-bucket-sandyaakevin-12345/ch1.txt"  # Ensure GCS bucket is accessible
    with open("/home/sandyaa2004/Try2/ch1.txt", "w") as f:
        f.write(extracted_text)
    
    await rag.upload_file(corpus_name=corpus.name, file_path=file_path)
    print("✅ Extracted text uploaded successfully.")

asyncio.run(upload_text_to_rag())

# Step 6: Retrieve Context from RAG
print("🔍 Running a RAG retrieval query...")
retrieval = rag.Retrieval(
    source=rag.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    ),
)

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="What made the woman in the control centre look at the narrator strangely?"
)

if response.contexts:
    print("🔎 RAG Response:", response.contexts[0].content)
else:
    print("❌ No relevant context found in the document. Try a different query.")

# Step 7: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")
print("💡 Generated Response:", gen_response.text)
