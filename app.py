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
    def __init__(self, openai_api_key, engine="gpt-3.5-turbo", temperature=0.5):
        import openai
        self.openai = openai
        self.openai.api_key = openai_api_key
        self.engine = engine
        self.temperature = temperature

    # Method for generating a completion with the OpenAI model
    def chat_completions(self, messages, **kwargs):
        response = self.openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            **kwargs
        )
        return response['choices'][0]['message']['content']

@app.route('/retrieve', methods=['POST'])
def retrieve_info():
    # Retrieve information from a collection
    collection_name = request.json.get("collection_name")
    query = request.json.get("query")
    chat_history = request.json.get("chat_history")
    if chat_history and isinstance(chat_history, str):  
        chat_history = json.loads(chat_history)

    chat_history = chat_history if chat_history else []

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(query, k=2)

    openai_api = OpenAI(openai_api_key=openai_api_key, temperature=0.2)
    
    # Start with the system message
    messages = [{"role": "system", "content": "You are a friendly, empathetic and helpful mental health coach. If the content found in the assistant messages is helpful and relevant in answering the users content then use it as context and knwoledge when answering the question in your own words. If the assistant message is not helpful or relevant you can disregard it and answer the user with your general knowledge and in your own words. You dont need to let the user know whether you found the assistant message helpful or not."}]

    # Add chat history if it exists
    if chat_history:
        messages.extend(chat_history)

    # Add the user```python
    # Add the user's message
    messages.append({"role": "user", "content": query})

    # Add the assistant messages
    for result in search_results:
        messages.append({"role": "assistant", "content": result.page_content})

    response = openai_api.chat_completions(messages, max_tokens=100, n=1, stop=None)

    return {"results": response}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
