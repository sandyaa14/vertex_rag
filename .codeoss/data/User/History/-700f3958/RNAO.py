import vertexai
from vertexai import rag

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Use the RAG corpus you just created
corpus_name = "projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856"  # Update with your corpus ID

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
try:
    rag.import_files(
        corpus_name,
        paths,
        transformation_config=transformation_config,  # Optional
        max_embedding_requests_per_min=1000,  # Optional
    )
    print("✅ File imported successfully into RAG Corpus!")
except Exception as e:
    print(f"❌ Error importing file: {e}")
