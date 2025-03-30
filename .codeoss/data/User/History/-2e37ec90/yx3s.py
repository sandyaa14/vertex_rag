from vertexai import rag
import vertexai

# Set up your project details
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Define the corpus name (use the same one from list_files.py)
corpus_name = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"

# Set up retrieval configuration
retrieval_config = rag.RagRetrievalConfig(top_k=3)  # Retrieve top 3 relevant results

# Query the RAG Corpus
query = "What is the main topic of the document?"
response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
    text=query,
    rag_retrieval_config=retrieval_config,
)

# Print the retrieved response
print("📄 Retrieved Response:")
print(response)
