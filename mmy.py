import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = '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'

persist_directory = "./storage"
pdf_directory = "build/pdfs/"  # Update this to your specific directory with PDFs

documents = []

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

print("Nuvita Baby Customer Support at your service! Ask me anything about our products.")
while True:
    user_input = input("Enter a query: ")
    if user_input.lower() == "exit":
        print("Thank you for using Nuvita Baby Customer Support. Goodbye!")
        break

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        print("Nuvita Baby Customer Support: ", llm_response["result"])
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
