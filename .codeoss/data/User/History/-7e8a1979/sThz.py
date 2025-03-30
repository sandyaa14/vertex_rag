import asyncio
import fitz  # PyMuPDF for PDF text extraction
import os
from google.cloud import storage
from vertexai.preview import rag
import vertexai

# Configuration
PROJECT_ID = "my-rag-project-455210"  
LOCATION = "us-central1"  
BUCKET_NAME = "rag-bucket-sandyaakevin-12345"
CORPUS_NAME = "my_rag_corpus"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

def process_pdf_input(input_path):
    """Process either a single PDF file or all PDFs in a folder."""
    pdf_texts = {}
    
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        # Single PDF file case
        try:
            doc = fitz.open(input_path)
            text = "\n".join(page.get_text() for page in doc)
            filename = os.path.basename(input_path)
            pdf_texts[filename] = text
            print(f"✅ Processed {filename} (pages: {len(doc)})")
        except Exception as e:
            print(f"❌ Error processing {input_path}: {str(e)}")
    elif os.path.isdir(input_path):
        # Folder case
        for filename in os.listdir(input_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(input_path, filename)
                try:
                    doc = fitz.open(file_path)
                    text = "\n".join(page.get_text() for page in doc)
                    pdf_texts[filename] = text
                    print(f"✅ Processed {filename} (pages: {len(doc)})")
                except Exception as e:
                    print(f"❌ Error processing {filename}: {str(e)}")
    else:
        print(f"❌ Invalid input path: {input_path}. Must be a PDF file or directory.")
    
    return pdf_texts

def upload_texts_to_gcs(texts_dict, bucket_name):
    """Upload extracted texts to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Ensure bucket exists
    if not bucket.exists():
        bucket.create(location=LOCATION)
        print(f"✅ Created bucket {bucket_name}")
    
    gcs_uris = []
    for filename, text in texts_dict.items():
        blob_name = f"pdf_texts/{filename.replace('.pdf', '.txt')}"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(text)
        gcs_uris.append(f"gs://{bucket_name}/{blob_name}")
        print(f"✅ Uploaded {filename} text to {blob_name}")
    
    return gcs_uris

async def create_and_populate_corpus(corpus_name, gcs_uris):
    """Create or get a RAG corpus and import files."""
    try:
        corpus = rag.get_corpus(name=corpus_name)
        print(f"✅ Using existing corpus: {corpus_name}")
    except Exception:
        corpus = rag.create_corpus(display_name=corpus_name)
        print(f"✅ Created new corpus: {corpus_name}")
    
    # Import files to corpus
    if gcs_uris:
        response = await rag.import_files_async(
            corpus_name=corpus.name,  # Changed from 'corpus' to 'corpus_name'
            gcs_source_uris=gcs_uris
        )
        print("✅ Files imported to corpus:", response)
    else:
        print("⚠️ No files to import to corpus")
    
    return corpus

async def query_corpus(corpus, question):
    """Query the RAG corpus with a question."""
    try:
        response = await rag.retrieval_query(
            corpus=corpus.name,
            query=question
        )
        
        print("\n🔍 Query Results:")
        if hasattr(response, "contexts") and response.contexts:
            for i, context in enumerate(response.contexts[:3]):  # Show top 3 contexts
                print(f"\n📄 Context {i+1} (Relevance: {context.relevance_score:.2f}):")
                print(context.text[:500] + "...")  # Show first 500 chars
        else:
            print("❌ No relevant context found.")
            
        if hasattr(response, "answer"):
            print(f"\n💡 Generated Answer:\n{response.answer}")
            
    except Exception as e:
        print(f"❌ Error querying corpus: {str(e)}")

async def main():
    print(f"🚀 Starting RAG setup for project {PROJECT_ID}")
    
    # Step 1: Process PDF input (file or folder)
    pdf_input = "/home/sandyaa2004/Try2/ch1.pdf"  # Can be a file or folder
    pdf_texts = process_pdf_input(pdf_input)
    
    if not pdf_texts:
        print("❌ No PDF files processed.")
        return
    
    # Step 2: Upload texts to GCS
    gcs_uris = upload_texts_to_gcs(pdf_texts, BUCKET_NAME)
    
    # Step 3: Create and populate RAG corpus
    corpus = await create_and_populate_corpus(CORPUS_NAME, gcs_uris)
    
    # Step 4: Interactive query loop
    print("\n🎯 Ready to query your documents. Type 'exit' to quit.")
    while True:
        question = input("\n❓ Enter your question: ").strip()
        if question.lower() == 'exit':
            break
        if question:
            await query_corpus(corpus, question)
        else:
            print("⚠️ Please enter a valid question.")

if __name__ == "__main__":
    asyncio.run(main())