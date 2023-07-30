import os
import requests
from flask import Flask, request
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.documents import Document


app = Flask(__name__)

api_key = os.getenv('OPENAI_API_KEY')

persist_directory = "./storage"
pdf_directory = "pdfs/"  # Update this to your specific directory with PDFs

documents = []

def get_shopify_products():
    url = "https://nuvitababy-com.myshopify.com/admin/api/2023-07/products.json"
    headers = {
        "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"
    }
    response = requests.get(url, headers=headers)
    products = response.json().get('products', [])
    # In this case, we are only interested in the product title and description
    # Create a Document object for each product
    return [Document(f'{product["title"]} {product["body_html"]}') for product in products]


# Get products from Shopify
documents.extend(get_shopify_products())

# Iterate over every file in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):  # If the file is a PDF
        pdf_path = os.path.join(pdf_directory, filename)  # Get the full path to the file
        loader = PyMuPDFLoader(pdf_path)
        document = loader.load()  # Load the document
        documents.extend(document)  # Add the document's content to the list

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model_name='gpt-4')  # Change model name to 'gpt-3.5-turbo' if you do not yet have access to 'gpt-4'

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


@app.route('/query', methods=['GET'])
def query():
    user_input = request.args.get('input', None) # Fetch the user input from the GET request
    if user_input is None:
        return {
            'status': 'error',
            'message': 'No input provided'
        }, 400

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        return {
            'status': 'success',
            'response': llm_response["result"]
        }
    except Exception as err:
        return {
            'status': 'error',
            'message': f'Exception occurred: {str(err)}'
        }, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Set the host and port to your needs
