import vertexai
from vertexai import preview
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import asyncio
import os

# 🔹 Set Google Cloud Project Details
PROJECT_ID = "my-rag-project-455210"  # Replace with your actual project ID
LOCATION = "us-central1"
BUCKET_NAME = "rag-bucket-sandyaakevin-12345"
CORPUS_NAME = "my_rag_corpus"

# 🔹 Set GCS Path for Your PDFs (Upload your files first!)
GCS_PATHS = ["gs://rag-bucket-sandyaakevin-12345/pdf_texts/ch1.txt"]

# 🔹 Authenticate (Ensure credentials are set in your environment)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account.json"  # Replace with your credentials file path

# 🔹 Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

async def main():
    print("🚀 Setting up RAG Corpus...")

    # 🔹 Create or Use Existing RAG Corpus
    embedding_model_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )

    backend_config = rag.RagVectorDbConfig(rag_embedding_model_config=embedding_model_config)

    try:
        corpora = rag.list_corpora()
        existing_corpus = next((c for c in corpora if c.display_name == CORPUS_NAME), None)

        if existing_corpus:
            print(f"✅ Using existing corpus: {CORPUS_NAME}")
            rag_corpus = existing_corpus
        else:
            print(f"📌 Creating new corpus: {CORPUS_NAME}")
            rag_corpus = rag.create_corpus(display_name=CORPUS_NAME, backend_config=backend_config)
    except Exception as e:
        print(f"❌ Error creating or fetching corpus: {e}")
        return

    # 🔹 Import Files into the RAG Corpus
    print("📂 Importing files into RAG Corpus...")
    transformation_config = rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
    )

    try:
        response = await rag.import_files_async(
            corpus_name=rag_corpus.name,
            paths=GCS_PATHS,
            transformation_config=transformation_config,
            max_embedding_requests_per_min=1000,
        )
        result = await response.result()
        print("✅ Files imported successfully:", result)
    except Exception as e:
        print(f"❌ Error importing files: {e}")
        return

    # 🔹 Setup RAG Retrieval
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=3,
        filter=rag.Filter(vector_distance_threshold=0.5)
    )

    # 🔹 Setup Retrieval Tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    )

    # 🔹 Setup Generative Model with RAG
    rag_model = GenerativeModel(
        model_name="gemini-1.5-flash-001",  # or "gemini-1.5-pro"
        tools=[rag_retrieval_tool]
    )

    # 🔹 Ask a Question
    print("\n🤖 Asking a question...")
    query = "What is RAG and why is it helpful?"
    try:
        response = rag_model.generate_content(query)
        print("💡 Answer:", response.text)
    except Exception as e:
        print(f"❌ Error generating answer: {e}")

if __name__ == "__main__":
    asyncio.run(main())
