from vertexai import rag

# Use the RAG corpus you just created
corpus_name = "projects/904559256331/locations/us-central1/ragCorpora/3458764513820540928"  # Update with your corpus ID

# Path to the file in your Cloud Storage bucket
paths = ["gs://rag-bucket-sandyaakevin-12345/ch1.pdf"]  # Replace with your actual bucket name

# Define chunking configuration
transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,  # Adjust chunk size as needed
        chunk_overlap=100,
    ),
)

# Import the PDF file into the corpus
rag.import_files(
    corpus_name,
    paths,
    transformation_config=transformation_config,  # Optional
    max_embedding_requests_per_min=1000,  # Optional
)

print("✅ File imported successfully into RAG Corpus!")
