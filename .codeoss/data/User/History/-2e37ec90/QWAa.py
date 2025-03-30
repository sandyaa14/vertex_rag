from vertexai.preview import rag
import vertexai

# ✅ Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"  # Replace with your actual project ID
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)

# ✅ Use the same RAG Corpus Name as before
RAG_CORPUS_NAME = "projects/904559256331/locations/us-central1/ragCorpora/3458764513820540928"

# ✅ Set up retrieval configuration
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=3,  # Retrieve top 3 most relevant results
    filter=rag.Filter(vector_distance_threshold=0.5)
)

# ✅ Query the RAG Corpus
query_text = "Summarize the content of the uploaded document."

response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=RAG_CORPUS_NAME,  
        )
    ],
    text=query_text,
    rag_retrieval_config=rag_retrieval_config,
)

# ✅ Print the Retrieved Context
print("\n🔹 Retrieved Context from RAG Corpus:")
print(response)
