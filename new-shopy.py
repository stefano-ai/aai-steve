import os
import requests
import langchain
from flask import Flask, request
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

api_key = os.getenv('OPENAI_API_KEY')

persist_directory = "./storage"
pdf_directory = "pdfs/"

shopify_documents = []
pdf_documents = []

class Document:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

def get_shopify_products():
    url = "https://nuvitababy-com.myshopify.com/admin/api/2023-07/products.json"
    headers = {
        "X-Shopify-Access-Token": "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"
    }
    page_info = None
    products = []
    
    while True:
        params = {"limit": 250}  # Get maximum number of products per request
        if page_info:
            params["page_info"] = page_info

        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        products.extend(data.get('products', []))
        
        link_header = response.headers.get("Link")
        if link_header:
            links = link_header.split(", ")
            next_link = [link for link in links if "rel=\"next\"" in link]
            if next_link:
                page_info = next_link[0].split("; ")[0].strip("<>").split("page_info=")[1]
            else:
                break
        else:
            break

    print("Fetched products from Shopify: ", len(products))
    return [f'{product["title"]} {product["body_html"]} {product["variants"][0]["price"]}' for product in products]

shopify_documents.extend(get_shopify_products())
print("Documents after adding Shopify products: ", len(shopify_documents))

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        loader = PyMuPDFLoader(pdf_path)
        document = loader.load()
        pdf_documents.extend(document)
        print(f"Added document: {filename}")

print("Total documents: ", len(shopify_documents) + len(pdf_documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
pdf_texts = [text.page_content for text in text_splitter.split_documents(pdf_documents)]
texts = shopify_documents + pdf_texts

print("Texts after splitting: ", len(texts))
print("First 5 texts: ", texts[:5])

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=[Document(text) for text in texts], 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name='gpt-4', temperature=1, max_tokens=1000)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route('/query', methods=['GET'])
def query():
    user_input = request.args.get('input', None)
    print(f"User input: {user_input}")
    
    if user_input is None:
        return {
            'status': 'error',
            'message': 'No input provided'
        }, 400

    query = f"###Prompt {user_input}"
    print(f"Query: {query}")
    
    try:
        llm_response = qa(query)
        print(f"LLM response: {llm_response}")
        return {
            'status': 'success',
            'response': llm_response["result"]
        }
    except Exception as err:
        print(f"Exception occurred: {str(err)}")
        return {
            'status': 'error',
            'message': f'Exception occurred: {str(err)}'
        }, 500

if __name__ == '__main__':
    print("Starting Flask application")
    app.run(host='0.0.0.0', port=8080)
