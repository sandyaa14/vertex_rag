import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio
import fitz  # PyMuPDF for PDF text extraction

# Step 1: Set up Google Cloud Project
PROJECT_ID = "my-rag-project-455210"  # Update with actual project ID
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

# Step 4: Extract text from PDF before importing it
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Local path for testing
pdf_path = "/home/sandyaa2004/Try2/ch1.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

if extracted_text:
    print("📜 Extracted PDF Text (First 500 chars):", extracted_text[:500])
else:
    print("⚠️ No text found in PDF. The document might be an image.")

# Step 5: Upload extracted text manually to RAG
document_content = rag.Document(content=extracted_text)
rag.create_document(corpus.name, document_content)
print("✅ Text document successfully added to RAG corpus.")

# Step 6: Retrieve Context from RAG
print("🔍 Running a RAG retrieval query...")
retrieval = rag.Retrieval(
    source=rag.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    ),
)

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus.name)],
    text="Summarize the content of the document."
)

# Step 7: Check if context retrieval works
if response.contexts:
    print("🔎 RAG Retrieved Contexts:")
    for i, context in enumerate(response.contexts):
        print(f"📌 Context {i+1}: {context.text}")
else:
    print("❌ No relevant context found in the document. Try a different query.")

# Step 8: Use RAG-Enhanced Generation with Gemini Model
rag_retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

gen_response = rag_model.generate_content("Explain Retrieval-Augmented Generation in simple terms.")
print("💡 Generated Response:", gen_response.text)
