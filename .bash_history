gcloud services enable aiplatform.googleapis.com storage.googleapis.com
gcloud config set project my-rag-project-455210
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
gcloud storage buckets create gs://my-rag-bucket --location=us-central1
gcloud storage buckets create gs://rag-bucket-sandyaakevin-12345 --location=us-central1
gcloud storage buckets list
gcloud storage cp ch1.pdf gs://rag-bucket-sandyaakevin-12345
gcloud storage ls gs://rag-bucket-sandyaakevin-12345
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-schema-uri=""     --project=my-rag-project-455210
echo '{"contentsDeltaUri": "gs://rag-bucket-sandyaakevin-12345/", "isIncremental": false}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
rm metadata.json
echo '{
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "COSINE"
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
rm metadata.json
echo '{
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "DOT_PRODUCT"
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
echo '{
  "config": {
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000
      }
    },
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "DOT_PRODUCT",
    "featureNormType": "UNIT_L2_NORM",
    "dimensions": 768
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
echo '{
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "DOT_PRODUCT",
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000
      }
    }
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
echo '{
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "dot_product",
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000
      }
    }
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
echo '{
  "config": {
    "dimensions": 768,
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "SQUARED_L2",
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000
      }
    }
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
gcloud components update
echo '{
  "config": {
    "algorithmConfig": {
      "treeAhConfig": {
        "leafNodeEmbeddingCount": 1000
      }
    },
    "approximateNeighborsCount": 10,
    "distanceMeasureType": "COSINE_DISTANCE",
    "featureNormType": "UNIT_L2_NORM",
    "dimensions": 768
  }
}' > metadata.json
gcloud ai indexes create     --display-name="my-rag-index"     --metadata-file=metadata.json     --project=my-rag-project-455210     --region=us-central1
python3 rag_setup.py
/bin/python /home/sandyaa2004/rag_setup.py
python3 rag_setup.py
gcloud storage buckets list
python3 import_files.py
python3 list_files.py
python3 retrieve_rag.py
nano list_files.py
python3 list_files.py
nano list_files.py
python3 list_files.py
nano list_files.py
python3 list_files.py
nano list_files.py
python3 list_files.py
python3 import_files.py
python3 list_files.py
gcloud ai rag-corpora list --project=my-rag-project-455210 --location=us-central1
gcloud components update
sudo apt-get update && sudo apt-get --only-upgrade install google-cloud-cli
gcloud ai rag-corpora list --project=my-rag-project-455210 --location=us-central1
python3 check_corpora.py
python3 list_files.py
python3 retrieve_rag.py
python3 list_files.py
python3 retrieve_rag.py
/bin/python /home/sandyaa2004/retrieve_rag.py
pip install --upgrade vertexai
pip install --upgrade google-cloud-aiplatform
python3 check_corpora.py
python3 retrieve_rag.py
nano retrieve_rag.py
python3 retrieve_rag.py
cat retrieve_rag.py
nano retrieve_rag.py
python3 retrieve_rag.py
python3 check_corpora.py
/bin/python /home/sandyaa2004/retrieve_rag.py
/bin/python /home/sandyaa2004/check_corpora.py
/bin/python /home/sandyaa2004/import_files.py
/bin/python /home/sandyaa2004/retrieve_rag.py
/bin/python /home/sandyaa2004/rag_setup.py
/bin/python /home/sandyaa2004/check_corpora.py
gcloud config get-value project
gcloud services enable aiplatform.googleapis.com
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
python rag_setup.py
python import_files.py
python list_files.py
python retrieve_rag.py
/bin/python /home/sandyaa2004/retrieve_rag.py
python retrieve_rag.py
/bin/python /home/sandyaa2004/retrieve_rag.py
python check_corpora.py
python list_files.py
pip show google-cloud-aiplatform
pip install --upgrade vertexai
python retrieve_rag.py
/bin/python /home/sandyaa2004/retrieve_rag.py
gcloud ai rag-corpora list-files --location=us-central1 --rag-corpus=6917529027641081856
gcloud ai rag-corpus list-files --location=us-central1 --rag-corpus=6917529027641081856
/bin/python /home/sandyaa2004/retrieve_rag.py
from vertexai.preview import rag
/bin/python /home/sandyaa2004/retrieve_rag.py
gsutil ls gs://rag-bucket-sandyaakevin-12345/
gcloud services enable aiplatform.googleapis.com
pip install google-cloud-aiplatform
python3 vertex_rag.py
/bin/python /home/sandyaa2004/Try2/vertex_rag.py
pip install --upgrade google-cloud-aiplatform
pip uninstall -y vertexai google-cloud-aiplatform
pip install vertexai google-cloud-aiplatform==1.71.1
python3
python3 -c "from vertexai import rag; print(dir(rag))"
pip show google-cloud-aiplatform vertexai
pip install --upgrade google-cloud-aiplatform vertexai
pip show google-cloud-aiplatform vertexai
pip uninstall -y google-cloud-aiplatform vertexai
pip install --upgrade google-cloud-aiplatform vertexai
pip uninstall -y google-cloud-aiplatform vertexai
pip install --no-cache-dir --upgrade google-cloud-aiplatform vertexai
pip install vertexai==  
/bin/python /home/sandyaa2004/Try2/sample.py
pip install --pre --upgrade google-cloud-aiplatform vertexai
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/new.py
python -c "import vertexai.preview.rag as rag; print(dir(rag))"
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/new1.py
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/new1.py
pip install pymupdf
/bin/python /home/sandyaa2004/Try2/new1.py
gsutil cp gs://rag-bucket-sandyaakevin-12345/ch1.pdf /home/sandyaa2004/Try2/
/bin/python /home/sandyaa2004/Try2/new1.py
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/new1.py
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/.another.py
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/vertex_rag.py
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/new1.py
/bin/python /home/sandyaa2004/Try2/sample.py
/bin/python /home/sandyaa2004/Try2/vertex_rag.py
gcloud storage cp ch2.docx gs://rag-bucket-sandyaakevin-12345
gcloud storage ls gs://rag-bucket-sandyaakevin-12345
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/kevinew
/bin/python /home/sandyaa2004/Try2/newtry
pip install google-cloud-ai
from google.cloud import aiplatform
pip install google-cloud-aiplatform
which python
/bin/python /home/sandyaa2004/Try2/newtry
/bin/python /home/sandyaa2004/Try2/kevinew
/bin/python /home/sandyaa2004/Try2/new.py
/bin/python /home/sandyaa2004/Try2/newtry
/bin/python /home/sandyaa2004/Try2/vertex_rag.py
pip install python-docx
/bin/python /home/sandyaa2004/Try2/vertex_rag.py
/bin/python /home/sandyaa2004/vertex_rag.py
git init
git add .
git status
git commit -m "Initial commit or message describing changes"
git config --global user.name "Sai Sandyaa"
git config --global user.email "sandyaa2004@gmail.com"
git init
rm -rf .git
git init
git add .
git config --global user.name "sandyaa14"
git config --global user.email "sandyaa2004@gmail..com"
git commit -m "Initial commit or message describing changes"
git remote add origin https://github.com/sandyaa14/vertex_rag.git
git push -u origin master
ssh-keygen -t rsa -b 4096 -C "sandyaa2004@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
rm -rf .git
git init
