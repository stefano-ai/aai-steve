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
pdf_directory = "pdfs/"  # Update this to your specific directory with PDFs

shopify_documents = []
pdf_documents = []

# Creating a simple class that mimics the document object expected by Chroma
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
        
        # Check if there is a link header for the next page
        link_header = response.headers.get("Link")
        if link_header:
            # Get the next page's page_info
            links = link_header.split(", ")
            next_link = [link for link in links if "rel=\"next\"" in link]
            if next_link:
                page_info = next_link[0].split("; ")[0].strip("<>").split("page_info=")[1]
            else:
                break
        else:
            break

    print("Fetched products from Shopify: ", len(products))  # Debugging
    return [f'{product["title"]} {product["body_html"]} {product["variants"][0]["price"]}' for product in products]

    url = "https://nuvitababy-com.myshopify.com/admin/api/2023-07/products.json"
    headers = {
        "X-Shopify-Access-Token": "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"
    }
    response = requests.get(url, headers=headers)
    products = response.json().get('products', [])
    print("Fetched products from Shopify: ", len(products))  # Debugging
    return [f'{product["title"]} {product["body_html"]} {product["variants"][0]["price"]}' for product in products]

shopify_documents.extend(get_shopify_products())
print("Documents after adding Shopify products: ", len(shopify_documents))  # Debugging

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):  # If the file is a PDF
        pdf_path = os.path.join(pdf_directory, filename)  # Get the full path to the file
        loader = PyMuPDFLoader(pdf_path)
        document = loader.load()  # Load the document
        pdf_documents.extend(document)  # Add the document's content to the list
        print(f"Added document: {filename}")  # Debugging

print("Total documents: ", len(shopify_documents) + len(pdf_documents))  # Debugging

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
pdf_texts = [text.page_content for text in text_splitter.split_documents(pdf_documents)]
texts = shopify_documents + pdf_texts  # Directly append Shopify descriptions to texts

print("Texts after splitting: ", len(texts))  # Debugging
print("First 5 texts: ", texts[:50])  # Print first 5 texts to check

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=[Document(text) for text in texts], 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name='gpt-4')  # Change model name to 'gpt-3.5-turbo' if you do not yet have access to 'gpt-4'

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route('/query', methods=['GET'])
def query():
    user_input = request.args.get('input', None)  # Fetch the user input from the GET request
    print(f"User input: {user_input}")  # Debugging
    if user_input is None:
        return {
            'status': 'error',
            'message': 'No input provided'
        }, 400

    query = f"###Prompt {user_input}"
    print(f"Query: {query}")  # Debugging
    try:
        llm_response = qa(query)
        print(f"LLM response: {llm_response}")  # Debugging
        return {
            'status': 'success',
            'response': llm_response["result"]
        }
    except Exception as err:
        print(f"Exception occurred: {str(err)}")  # Debugging
        return {
            'status': 'error',
            'message': f'Exception occurred: {str(err)}'
        }, 500


if __name__ == '__main__':
    print("Starting Flask application")  # Debugging
    app.run(host='0.0.0.0', port=8080)  #
