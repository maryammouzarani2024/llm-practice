
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
openai = OpenAI()

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
        
        # Return the transcribed text
        return transcript.text
    
    except Exception as e:
        return f"Error during transcription: {str(e)}"

# Create a Gradio interface with a more streamlined experience
with gr.Blocks() as demo:
    gr.Markdown("Hold the microphone button to record your voice, release to stop. Transcription will start automatically.")
    
    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",
            label="Hold to Record",
            sources=["microphone"],
            streaming=False,
            interactive=True
        )
    
    with gr.Row():
        text_output = gr.Textbox(label="Transcription", lines=5)
    
    # This is the key part - automatically call the transcribe function when audio recording is done
    audio_input.stop_recording(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[text_output]
    )

# Launch the interface
demo.launch()