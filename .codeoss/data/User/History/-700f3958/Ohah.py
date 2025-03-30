import vertexai
from vertexai import rag
import subprocess

# Initialize Vertex AI
PROJECT_ID = "my-rag-project-455210"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Get the first available corpus dynamically
try:
    corpora = list(rag.list_corpora())  # Convert pager to a list
    if not corpora:
        raise ValueError("❌ No RAG corpus found. Please create one first.")
    
    corpus_name = corpora[0].name
    print(f"✅ Using RAG Corpus: {corpus_name}")

except Exception as e:
    print(f"❌ Error fetching corpora: {e}")
    exit()

# Define the file path in Cloud Storage
GCS_PATH = "gs://rag-bucket-sandyaakevin-12345/ch1.pdf"

# Check if the file exists in GCS
check_file = subprocess.run(["gsutil", "ls", GCS_PATH], capture_output=True, text=True)

if "CommandException" in check_file.stderr:
    print(f"❌ File not found in GCS: {GCS_PATH}")
    exit()
else:
    print(f"✅ File found in GCS: {GCS_PATH}")

# Define chunking configuration
transformation_config = rag.TransformationConfig(
    chunking_config=rag.ChunkingConfig(
        chunk_size=512,
        chunk_overlap=100,
    ),
)

# Import the PDF file into the corpus
try:
    rag.import_files(
        corpus_name,
        [GCS_PATH],
        transformation_config=transformation_config,
        max_embedding_requests_per_min=1000,
    )
    print("✅ File imported successfully into RAG Corpus!")
except Exception as e:
    print(f"❌ Error importing file: {e}")
