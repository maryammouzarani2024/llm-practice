import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
openai = OpenAI()

# Functions for RAG implementation
def read_pdf(file_path):
    """Read a PDF file and extract text content."""
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def load_knowledge_base():
    """Load all PDFs from the knowledge-base directory and create a vector store."""
    # Create the knowledge-base directory if it doesn't exist
    os.makedirs("knowledge-base", exist_ok=True)
    
    # Get all PDF files in the knowledge-base directory
    pdf_files = glob.glob("knowledge-base/*.pdf")
    
    if not pdf_files:
        print("No PDF files found in the knowledge-base directory.")
        return None
    
    # Read and concatenate all PDF content
    all_content = ""
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        content = read_pdf(pdf_file)
        all_content += content + "\n\n"
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_content)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    print(f"Created vector store with {len(chunks)} chunks from {len(pdf_files)} PDF files")
    return vector_store

# Initialize vector store
vector_store = load_knowledge_base()
if vector_store:
    # Create retrieval chain
    llm = ChatOpenAI(model=MODEL)
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    print("RAG system initialized successfully")
else:
    print("RAG system not initialized. Please add PDF files to the knowledge-base directory.")
    retrieval_chain = None



#audio generation
    
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
def talker(message):
        response=openai.audio.speech.create(
            
            model="tts-1",
            voice="onyx",
            input=message
        )
        audio_stream=BytesIO(response.content)
        audio=AudioSegment.from_file(audio_stream, format="mp3")
        play(audio)

system_message = "You are a helpful assistant for tourists visiting a city."
system_message += "Help the user and give him or her good explanation about the cities or places."
system_message += "Talk about history, geography and current conditions."
system_message += "Start with a short explanation about three lines and when the user wants explain more."
system_message += "Use the retrieved information from knowledge base when available to give detailed and accurate information."

#gradio handles the history of user messages and the assistant responses

def chat(history):
    # Extract just the content from the message history for RAG
    chat_history = []
    messages = [{"role": "system", "content": system_message}]
    
    for i in range(0, len(history), 2):
        if i+1 < len(history):
            user_msg = history[i]["content"]
            ai_msg = history[i+1]["content"] if i+1 < len(history) else ""
            chat_history.append((user_msg, ai_msg))
            messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})
    
    # Get the latest user message
    latest_user_message = history[-1]["content"] if history and history[-1]["role"] == "user" else ""
    
    # Use RAG if available, otherwise use the standard OpenAI API
    if retrieval_chain and latest_user_message:
        try:
            rag_response = retrieval_chain.invoke({
                "question": latest_user_message,
                "chat_history": chat_history[:-1] if chat_history else []
            })
            reply = rag_response["answer"]
            print(reply)
        except Exception as e:
            print(f"Error using RAG: {str(e)}")
            # Fallback to standard API
            response = openai.chat.completions.create(model=MODEL, messages=messages)
            reply = response.choices[0].message.content
    else:
        # Standard OpenAI API
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
    
    history += [{"role":"assistant", "content":reply}]
    talker(reply)
    
    return history

def transcribe_audio(audio_path):
   
    try:
        # Check if audio_path is valid
        if audio_path is None:
            return "No audio detected. Please record again."
        
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
             transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        return transcript.text
    
    except Exception as e:
        return f"Error during transcription: {str(e)}"




##################Interface with Gradio##############################

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"]
)

# Load CSS from external file
with open('style.css', 'r') as f:
    css = f.read()

def refresh_knowledge_base():
    """Reload the knowledge base and update the retrieval chain."""
    global vector_store, retrieval_chain
    
    vector_store = load_knowledge_base()
    if vector_store:
        # Create retrieval chain
        llm = ChatOpenAI(model=MODEL)
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        return "Knowledge base refreshed successfully!"
    else:
        return "No PDF files found in the knowledge-base directory."

with gr.Blocks(theme=theme, css=css) as ui:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 🌍 Tourist Assistant", elem_classes="title")
        gr.Markdown("Ask about any city, landmark, or destination around the world", elem_classes="subtitle")
        
        with gr.Blocks() as demo:
            gr.Image("travel.jpg", show_label=False, height=150, container=False, interactive=False)
 
        with gr.Column(elem_classes="chatbot-container"):
            chatbot = gr.Chatbot(
                height=400, 
                type="messages",
                bubble_full_width=False,
                show_copy_button=True,
                elem_id="chatbox"
            )
        
        with gr.Row():
            entry = gr.Textbox(
                label="",
                placeholder="Type your question here or use the microphone below...",
                container=False,
                lines=2,
                scale=10
            )
            
            with gr.Column(scale=1, elem_classes="clear-button"):
                clear = gr.Button("Clear", variant="secondary", size="sm")
        
        with gr.Row():
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Knowledge Base", variant="primary", size="sm")
                refresh_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row(elem_classes="mic-container"):
            audio_input = gr.Audio(
                type="filepath",
                label="🎤 Record",
                sources=["microphone"],
                streaming=False,
                interactive=True,
                autoplay=False,
                show_download_button=False,
                show_share_button=False,
                elem_id="mic-button"
            )
        
            
    def transcribe_and_submit(audio_path):
        transcription = transcribe_audio(audio_path)
        history = chatbot.value if chatbot.value else []
        history += [{"role":"user", "content":transcription}]
        return transcription, history, history, None
        
    audio_input.stop_recording(
        fn=transcribe_and_submit,
        inputs=[audio_input],
        outputs=[entry, chatbot, chatbot, audio_input]
    ).then(
        chat, inputs=chatbot, outputs=[chatbot]
    )

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)
    refresh_btn.click(refresh_knowledge_base, inputs=None, outputs=refresh_status)

ui.launch(inbrowser=True)