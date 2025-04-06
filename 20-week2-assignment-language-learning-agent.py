import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


# Initialization

load_dotenv(override=True)

# Access openai api key using os.getenv
api_key = os.getenv('OPENAI_API_KEY')


# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    


   
MODEL = "gpt-4o-mini"
openai = OpenAI()




#image creation
import base64
from io import BytesIO
from PIL import Image
import openai
  
def artist(language):
    image_response=openai.images.generate(
        model="dall-e-3",
        prompt=f"an image representing someone in a cpuntry with language {language} and learns the language",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    
    image_base64=image_response.data[0].b64_json
    image_data=base64.b64decode(image_base64) #decode image into bytes
    return Image.open(BytesIO(image_data))


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
        
#flight agent code

system_message = "You are a helpful assistant for a language learning center. "
system_message += "Help the user to have a conversation in the language he or she wants "
system_message += "Always be accurate. If you don't know the answer, say so."

# a helpful function about the ticket price to differenct destinations:

language_prices = {"English": "$799", "German": "$899", "French": "$1400", "Spanish": "$499"}

def get_course_price(language):
    print(f"Tool get_course_price called for {language}")
    city = language.lower()
    return language_prices.get(language, "Unknown")

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_course_price",
    "description": "Get the price of a language course. Call this whenever you need to know the ticket price, for example when a customer asks 'How much costs a language course '",
    "parameters": {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "description": "The language the customer wants to learn",
            },
        },
        "required": ["language"],
        "additionalProperties": False
    }
}

# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]



def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    # Comment out or delete the next line if you'd rather skip Audio for now..
    talker(reply)
    
    return history, image

# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    language = arguments.get('language')
    price = get_course_price(language)
    response = {
        "role": "tool",
        "content": json.dumps({"language": language,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, language

# More involved Gradio code as we're not using the preset Chat interface!
# Passing in inbrowser=True in the last line will cause a Gradio window to pop up immediately.

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

ui.launch(inbrowser=True)