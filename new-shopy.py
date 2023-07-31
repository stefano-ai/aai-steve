import os
import requests
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

pdf_documents = []

class Document:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        loader = PyMuPDFLoader(pdf_path)
        document = loader.load()
        pdf_documents.extend(document)
        print(f"Added document: {filename}")

print("Total documents: ", len(pdf_documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
pdf_texts = [text.page_content for text in text_splitter.split_documents(pdf_documents)]

print("Texts after splitting: ", len(pdf_texts))
print("First 5 texts: ", pdf_texts[:5])

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=[Document(text) for text in pdf_texts], 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # Returns the top 10 most similar documents
llm = ChatOpenAI(model_name='gpt-4-0613', temperature=1, max_tokens=1000)

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
