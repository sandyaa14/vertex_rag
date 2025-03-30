import vertexai.preview.rag as rag
print(dir(rag))

import vertexai
from vertexai.generative_models import GenerativeModel, Tool

# Set Google Cloud Project Details
PROJECT_ID = "my-rag-project-455210"  # Replace with your project ID
LOCATION = "us-central1"
CORPUS_ID = "6917529027641081856"  # Your RAG Corpus ID

# Initialize Vertex AI API
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Function to check if corpus exists
def check_corpus():
    corpora = list(rag.list_corpora())
    for corpus in corpora:
        print(f"Found Corpus: {corpus.name}")
        if CORPUS_ID in corpus.name:
            return True
    return False

if not check_corpus():
    raise ValueError("RAG Corpus not found! Ensure it is created before running retrieval.")

# Get corpus name
corpus_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{CORPUS_ID}"

# Function to check if file exists in RAG
def check_files():
    files = list(rag.list_files(corpus_name))
    if not files:
        print("No files found in RAG. Ensure your document is uploaded and processed.")
        return False
    for file in files:
        print(f"📄 Found File: {file.name}")
    return True

if not check_files():
    raise ValueError("Document not found in RAG. Re-upload the file and retry.")

# Retrieve document contents
def retrieve_content(query):
    print(f"\n Querying RAG with: {query}\n")
    
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=3,  # Get top 3 results
        filter=rag.Filter(vector_distance_threshold=0.5)  # Ensure relevant retrieval
    )

    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query,
        rag_retrieval_config=rag_retrieval_config,
    )
    
    if response:
        print("Retrieved Text from RAG:")
        for candidate in response.candidates:
            print(candidate.content.parts[0].text)  # Extract text
    else:
        print("No relevant content found. Check if document contains text data.")

# Run Retrieval
retrieve_content("What is the document about?")
