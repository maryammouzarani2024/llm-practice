
import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
import openai
import google.generativeai
import anthropic

import gradio as gr 

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
    
    
system_message = "You are a helpful assistant"

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content


message_gpt("What is today's date?")

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

#to call gradio pass the function name and define the input and output source:
#share=true, means that gradio builds a public url for this page
gr.Interface(fn=shout,  inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)], flagging_mode="never").launch(share=True)