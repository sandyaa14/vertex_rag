from vertexai.preview.rag import (
    RagCorpus,
    RagFile,
    Retrieval,
    retrieval_query,
    list_corpora,
    upload_file
)

def main():
    # ✅ Step 1: List available corpora
    print("📌 Listing existing corpora...")
    corpora = list_corpora()
    if corpora:
        print(f"✅ Existing corpora: {[c.display_name for c in corpora]}")
        corpus = corpora[0]  # Use first available corpus
    else:
        # ✅ Step 2: Create a new corpus if none exist
        print("🚀 No existing corpora found. Creating a new one...")
        corpus = RagCorpus.create_corpus(display_name="My Knowledge Base")
        print(f"✅ Created corpus: {corpus.display_name}")

    # ✅ Step 3: Upload a file to the corpus
    file_path = "example.txt"  # Update with your file path
    print(f"📂 Uploading file: {file_path} to corpus: {corpus.display_name}...")
    rag_file = upload_file(file_path=file_path, corpus=corpus)
    print(f"✅ Uploaded file ID: {rag_file.name}")

    # ✅ Step 4: Perform a retrieval query
    query = "What is Vertex AI RAG?"
    print(f"🔍 Querying: {query}...")
    response = retrieval_query(query=query)
    print(f"🤖 AI Response: {response}")

if __name__ == "__main__":
    main()
