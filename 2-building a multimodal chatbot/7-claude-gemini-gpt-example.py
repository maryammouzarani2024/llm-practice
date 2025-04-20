#Here we ask a simple question from different llms as a showcase of working with them

#load the api keys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")
    
if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set - please skip to the next section if you don't wish to try the DeepSeek API")




from openai import OpenAI
import anthropic
import google.generativeai

#setting the prompts
system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

def gpt_3_5_turbo():
    completion = openai.chat.completions.create(model='gpt-3.5-turbo', messages=prompts)
    print(completion.choices[0].message.content)
    

def gpt_4o_mini():
    completion = openai.chat.completions.create(
    model='gpt-4o-mini',
    messages=prompts,
    temperature=0.7
    )
    print(completion.choices[0].message.content)
    
def gpt_4o():
    
    completion = openai.chat.completions.create(
    model='gpt-4o',
    messages=prompts,
    temperature=0.4
    )
    print(completion.choices[0].message.content)
    

def claude_3_5_sonnet():
    claude = anthropic.Anthropic()
    
    message = claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=200,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )

    print(message.content[0].text)
    
    
def claude_3_5_sonnet_stream():
    message = claude.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
    )


    with result as stream:
        for text in stream.text_stream:
                clean_text = text.replace("\n", " ").replace("\r", " ")
                print(clean_text, end="", flush=True)
                
def gemini_with_google_library():
    google.generativeai.configure()
    gemini = google.generativeai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    system_instruction=system_message
    )
    response = gemini.generate_content(user_prompt)
    print(response.text)
    
               
def gemini_with_openai_client_library():
    gemini_via_openai_client = OpenAI(
        api_key=google_api_key, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    response = gemini_via_openai_client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        messages=prompts
    )
    print(response.choices[0].message.content)
    
def deepseek_model():
    
        
    deepseek_via_openai_client = OpenAI(
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com"
        )

    response = deepseek_via_openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=prompts,
    )

    print(response.choices[0].message.content)            

    

# print("testing claude LLM:")
# claude_3_5_sonnet()

# print("testing deepseek LLM:")
# deepseek_model()

# print("testing google gemini LLM with openai library:")
# gemini_with_openai_client_library()

# print("testing google gemini with google library:")
# gemini_with_google_library()

print("testing claude LLM:")
claude_3_5_sonnet()