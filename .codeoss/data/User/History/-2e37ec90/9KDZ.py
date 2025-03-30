Found Corpus: projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856
📄 Found File: projects/904559256331/locations/us-central1/ragCorpora/6917529027641081856/ragFiles/5400227338533536385

🔎 Querying RAG with: What is the document about?

Traceback (most recent call last):
  File "/home/sandyaa2004/retrieve_rag.py", line 65, in <module>
    retrieve_content("What is the document about?")
  File "/home/sandyaa2004/retrieve_rag.py", line 46, in retrieve_content
    rag_retrieval_config = rag.RagRetrievalConfig(
                           ^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'vertexai.preview.rag' has no attribute 'RagRetrievalConfig'
sandyaa2004@cloudshell:~$ ^C
sandyaa2004@cloudshell:~$ /bin/python /home/sandyaa2004/retrieve_rag.py
['EmbeddingModelConfig', 'JiraQuery', 'JiraSource', 'Pinecone', 'RagCorpus', 'RagFile', 'RagManagedDb', 'RagResource', 'Retrieval', 'SharePointSource', 'SharePointSources', 'SlackChannel', 'SlackChannelsSource', 'VertexFeatureStore', 'VertexRagStore', 'VertexVectorSearch', 'Weaviate', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'create_corpus', 'delete_corpus', 'delete_file', 'get_corpus', 'get_file', 'import_files', 'import_files_async', 'list_corpora', 'list_files', 'rag_data', 'rag_retrieval', 'rag_store', 'retrieval_query', 'update_corpus', 'upload_file', 'utils']
Found Corpus: projects/my-rag-project-455210/locations/us-central1/ragCorpora/6917529027641081856
📄 Found File: projects/904559256331/locations/us-central1/ragCorpora/6917529027641081856/ragFiles/5400227338533536385

🔎 Querying RAG with: What is the document about?

Traceback (most recent call last):
  File "/home/sandyaa2004/retrieve_rag.py", line 66, in <module>
    retrieve_content("What is the document about?")
  File "/home/sandyaa2004/retrieve_rag.py", line 47, in retrieve_content
    rag_retrieval_config = rag.RagRetrievalConfig(
                           ^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'vertexai.preview.rag' has no attribute 'RagRetrievalConfig'
sandyaa2004@cloudshell:~$ 