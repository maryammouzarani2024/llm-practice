#In this code, we ask two LLMs to explain what does a line of code do and why
#We use GPT-4o-mini and llama3.2

#setting up GPT-4o-mini

#set up LLM
from openai import OpenAI
from dotenv import load_dotenv
import os
 
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
GPT_MODEL = 'gpt-4o-mini'
openai = OpenAI()



#setting up llama
LLAMA_MODEL='llama3.2'



#prepare the prompts
system_prompt = "You are an assistant that analyzes codes and explain what a code does in details"

def create_user_prompt(code):
    user_prompt = f"Please explain what does this code do and why: \n"
    user_prompt +=code
    return user_prompt

def create_prompts(code):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": create_user_prompt(code)}
    ]




def ask_llama(code):
    ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    stream = ollama_via_openai.chat.completions.create(
        model=LLAMA_MODEL,
        messages=create_prompts(code),
        stream=True  # Enable streaming here

    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or '', end='')
    
    
def ask_openai(code):
    stream = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=create_prompts(code),
        stream=True
    )
    
    for chunk in stream:
        print(chunk.choices[0].delta.content or '', end='')
        
        
code="""yield from {book.get("author") for book in books if book.get("author")}"""

print("asking gpt-4o-mini")
ask_openai(code)


print("asking llama")
ask_llama(code)

    

