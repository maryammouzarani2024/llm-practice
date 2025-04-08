import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

#ollama is already installed in the local machine
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"

# Create a messages list using the same format that we used for OpenAI

messages = [
    {"role": "user", "content": "Please explain how I can learn python better"}
]
def ollama_api_direct_call():
        
    #prepare the json payload for the api post request
    payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False
        }

    response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
    print(response.json()['message']['content'])
    
    
def ollama_chat_interface_call():
    from openai import OpenAI
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    response = ollama_via_openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    print(response.choices[0].message.content)


def ollama_deepseek_model_call():
    from openai import OpenAI
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

    response = ollama_via_openai.chat.completions.create(
    model="deepseek-r1:1.5b",
    messages=[{"role": "user", "content": "Please give definitions of some core concepts behind LLMs: a neural network, attention and the transformer"}]
    )

    print(response.choices[0].message.content)



print("Calling Ollama's API directly: ")    
ollama_api_direct_call()

print("Calling Ollama's using the chat interface in openai library: ")
ollama_chat_interface_call()

print("Calling ollama's deepseek model:")
print("Make sure that you have pulled this model before via  'ollama pull deepseek-r1:1.5b' ")
ollama_deepseek_model_call()