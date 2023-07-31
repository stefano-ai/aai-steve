import os
from flask import Flask, request
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from threading import Lock

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

retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Returns the top 10 most similar documents
llm = ChatOpenAI(model_name='gpt-4-0613', temperature=1, max_tokens=1000)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Create a dictionary to store the conversation context for each user
context_dict = {}

# Create a lock to protect the conversation context
context_lock = Lock()

@app.route('/query', methods=['GET'])
def query():
    user_input = request.args.get('input', None)
    user_id = request.args.get('userid', None) # You should get a unique identifier for each user
    print(f"User input: {user_input}")
    
    if user_input is None or user_id is None:
        return {
            'status': 'error',
            'message': 'No input or userid provided'
        }, 400

    # Retrieve the user's context if it exists, otherwise create a new context
    with context_lock:
        user_context = context_dict.get(user_id, [])

    # Add the new input to the user's context
    user_context.append(user_input)

    query = f"###System: Sei un agente del servizio clienti su Nuvitababy.com, se non conosci la risposta, suggerisci ai clienti di andare sulla pagina di supporto del sito web per inviare una richiesta al supporto clienti.###Prompt: {' '.join(user_context)}"
    print(f"Query: {query}")

    # Identify the user intent
    user_intent = identify_user_intent(user_input)
    print(f"User intent: {user_intent}")

    try:
        llm_response = qa(query)
        print(f"LLM response: {llm_response}")

        # Add the model's response to the user's context
        user_context.append(llm_response["result"])
        
        # Update the user's context
        # Update the user's context in the dictionary
        with context_lock:
            context_dict[user_id] = user_context
        
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

def identify_user_intent(user_input):
    # TODO: Implement this function to identify the user intent
    return "unknown"
if __name__ == '__main__':
    print("Starting Flask application")
    app.run(host='0.0.0.0', port=8080)
