from flask import Flask, request
from flask_cors import CORS
import json
import os
from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Loading environment variables
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')
cohere_api_key = os.environ.get('cohere_api_key')
qdrant_url = os.environ.get('qdrant_url')
qdrant_api_key = os.environ.get('qdrant_api_key')

#Flask config
app = Flask(__name__)
CORS(app)

# Test default route
@app.route('/')
def hello_world():
    return {"Hello":"World"}

@app.route('/embed', methods=['POST'])
def embed_pdf():
    # Embedding code
    collection_name = request.json.get("collection_name")
    file_url = request.json.get("file_url")

    loader = PyPDFLoader(file_url)
    docs = loader.load_and_split()
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    
    return {"collection_name":qdrant.collection_name}

# Create a class for OpenAI interactions
class OpenAI:
    def __init__(self, openai_api_key, engine="text-davinci-002", temperature=0.7):
        import openai
        self.openai = openai
        self.openai.api_key = openai_api_key
        self.engine = engine
        self.temperature = temperature

    # Method for generating a completion with the OpenAI model
    def chat_completions(self, messages, **kwargs):
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        completions = self.openai.Completion.create(engine=self.engine, prompt=prompt, **kwargs)
        return completions.choices[0].text.strip()

@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    # Retrieve information from a collection
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")
    chat_history = request.json.get("chat_history")
    if isinstance(chat_history, str):
        chat_history = json.loads(chat_history)

    chat_history = chat_history if chat_history else []

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(query, k=2)

    openai_api = OpenAI(openai_api_key=openai_api_key, temperature=0.2)
    
    # Concatenate the chat history and the new query and search results
    messages = chat_history + [{"role": "user", "content": query}]
    for result in search_results:
        messages.append({"role": "assistant", "content": result.page_content})

    response = openai_api.chat_completions(messages, max_tokens=100, n=1, stop=None)

    return {"results": response}

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)

