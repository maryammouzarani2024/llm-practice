
import os
import glob 
from dotenv import load_dotenv

import gradio as gr

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder


MODEL = "gpt-4o-mini"
db_name = "vector_db"


def llm_setup(input_openai_key):
     # Load environment variables from .env file
    load_dotenv()
    if input_openai_key: 
        api_key=input_openai_key
    else:
        # Access openai api key using os.getenv
        api_key = os.getenv('OPENAI_API_KEY') 

    # Check the key

    if not api_key :
        return "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!",None
    else: 
            
        conversation_chain=llm_chat_setup(vectorization(read_pdfs()))
        return "API key is accepted!", conversation_chain
    
      
def read_pdfs():
    documents = []
    folders = glob.glob("knowledge-base/*")
    for folder in folders:
        doc_type = os.path.basename(folder)
        # Load all PDFs recursively in this folder
        loader = DirectoryLoader(
            folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader  # this handles PDF parsing
        )
        folder_docs = loader.load()
        # Add metadata to each document
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


def vectorization(documents):
    
    #split the documents into chunks using longchain characterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200) #the chunks should have some overlaps so that we will not miss parts of info in the beginning or at the end of a chunk
    chunks = text_splitter.split_documents(documents)

    #check metadata for each chunk:
    doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
    print(f"Document types found: {', '.join(doc_types)}")

    #vectorization:
    # Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
    #using OpenAI embeddings:
    embeddings = OpenAIEmbeddings()
    # Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()       
    # Create our Chroma vectorstore from the chunks
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore


def llm_chat_setup(vectorstore):
 
  # create a new Chat with OpenAI
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    
    # the retriever is an abstraction over the VectorStore
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def chat(message, history, chain):
    if chain is None:
        return "⚠️ Please set your OpenAI API key first."

    result = chain.invoke({"question": message})
    return result["answer"]



with gr.Blocks() as view:
    gr.Markdown("## 🔐 Set Your OpenAI API Key")
    with gr.Row():
        api_input = gr.Textbox(placeholder="Enter OpenAI API key", type="password", show_label=False)
        set_button = gr.Button("Set API Key")
        status_output = gr.Textbox(label="Status", interactive=False)

        chain_state = gr.State()  # Shared memory for conversation_chain

        set_button.click(
            fn=llm_setup,
            inputs=api_input,
            outputs=[status_output, chain_state]
        )

    gr.Markdown("## 💬 Chat with your RAG Assistant")
    
    

    gr.ChatInterface(fn=lambda msg, hist, chain: chat(msg, hist, chain), 
                     additional_inputs=[chain_state], 
                     type="messages")



# Launch the app in the browser
view.launch(inbrowser=True)