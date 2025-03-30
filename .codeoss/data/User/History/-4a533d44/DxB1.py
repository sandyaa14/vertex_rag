import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio
import fitz  # PyMuPDF
from google.cloud import storage
import os

# Configuration
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
BUCKET_NAME = "rag-bucket-sandyaakevin-12345"
CORPUS_NAME = "my_rag_corpus"
PDF_PATH = "/home/sandyaa2004/Try2/ch1.pdf"  # Your local PDF path

# Initialize Vertex AI and Storage
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    storage_client = storage.Client()
except Exception as e:
    print(f"❌ Error initializing Vertex AI or Storage: {e}")
    exit()

class PDFRAGSystem:
    def __init__(self):
        self.corpus = None
        self.retrieval_tool = None
        self.model = None

    def upload_pdf_to_gcs(self):
        """Upload PDF text to GCS and return URI"""
        try:
            text = self.extract_text_from_pdf(PDF_PATH)
            if not text:
                return None

            bucket = storage_client.lookup_bucket(BUCKET_NAME)
            if not bucket:
                bucket = storage_client.create_bucket(BUCKET_NAME, location=LOCATION)
                print(f"✅ Created bucket {BUCKET_NAME}")

            blob_name = os.path.basename(PDF_PATH).replace('.pdf', '.txt')
            blob = bucket.blob(f"pdf_texts/{blob_name}")
            blob.upload_from_string(text)
            gcs_uri = f"gs://{BUCKET_NAME}/pdf_texts/{blob_name}"
            print(f"✅ Uploaded text to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            print(f"❌ Error uploading to GCS: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            print(f"✅ Extracted text from {os.path.basename(pdf_path)} (pages: {len(doc)})")
            return text
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return None

    async def setup_rag_corpus(self):
        """Create or get RAG corpus and import files"""
        try:
            corpora = rag.list_corpora()
            existing_corpus = next((c for c in corpora if c.display_name == CORPUS_NAME), None)
            
            if existing_corpus:
                print(f"✅ Using existing corpus: {CORPUS_NAME}")
                self.corpus = existing_corpus
            else:
                print(f"📌 Creating new corpus: {CORPUS_NAME}")
                self.corpus = rag.create_corpus(display_name=CORPUS_NAME)

            gcs_uri = self.upload_pdf_to_gcs()
            if not gcs_uri:
                return False

            print("📂 Importing files into RAG Corpus...")
            from vertexai.preview.rag import RagFile, GcsSource
            rag_file = RagFile(gcs_source=GcsSource(uris=[gcs_uri]))
            
            await rag.import_files_async(
                corpus=self.corpus,
                files=[rag_file]
            )
            print("✅ Files imported successfully.")
            return True
        except Exception as e:
            print(f"❌ Error setting up RAG corpus: {e}")
            return False

    def initialize_model(self):
        """Initialize the Gemini model with RAG retrieval"""
        try:
            retrieval = rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=self.corpus)],
                ),
            )
            self.retrieval_tool = Tool.from_retrieval(retrieval=retrieval)
            self.model = GenerativeModel(
                model_name="gemini-1.5-flash-001",
                tools=[self.retrieval_tool]
            )
            print("✅ Model initialized with RAG capabilities")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")

    def ask_question(self, question):
        """Query the RAG system with a question"""
        try:
            if not self.model:
                print("❌ Model not initialized")
                return None

            print(f"\n🤖 Question: {question}")
            response = self.model.generate_content(question)
            answer = response.candidates[0].text if response.candidates else "No response."
            print(f"💡 Answer: {answer}")
            return answer
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return None

async def main():
    print("🚀 Starting PDF RAG System Setup")
    rag_system = PDFRAGSystem()
    
    if not await rag_system.setup_rag_corpus():
        print("❌ Failed to setup RAG corpus")
        return
    
    rag_system.initialize_model()
    
    print("\n🎯 Ready to answer questions about your PDF. Type 'exit' to quit.")
    while True:
        question = input("\n❓ Enter your question: ").strip()
        if question.lower() == 'exit':
            break
        if question:
            rag_system.ask_question(question)
        else:
            print("⚠️ Please enter a valid question.")

if __name__ == "__main__":
    asyncio.run(main())
