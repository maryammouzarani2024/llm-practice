#This sample code, gets a piece of python code translates it into c++ and optimizes it 
#We compare two models, gpt and claude in this work


import os
import requests
from dotenv import load_dotenv
import openai
import anthropic
import gradio as gr
import sys, io, subprocess

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
    
  
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")



openai = openai.OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"


system_message = "You are an assistant that analyzes a c or c++ code and finds its possible security vulnerabilities."
system_message += "Just accept c or c++ codes. You have to create an equivalent code in the same language and mitigate the detected vulnerabilities. "
system_message += "The C++ response needs to produce an identical output. Write a comment at the places you changed to show the previous version and possible vulnerability that you removed in the new code."


def user_prompt_for(c_code):
    user_prompt = "Improve this code and remove its possible vulnerability."
    user_prompt += "Respond in the same language; do not explain your work other than a few comments. "
    user_prompt += "Possible vulnerabilities are buffer overflow, memory leakage, heap errors and so on.\n\n"
    user_prompt += "Write anything other than the code as comment in your reponse.\n\n"
    user_prompt += c_code
    return user_prompt

def messages_for(c_code):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(c_code)}
    ]
    
    
def write_output(cpp):
    code = cpp.replace("```cpp","").replace("```","")
    with open("secured.cpp", "w") as f:
        f.write(code)

#stream the llm outputs      
def stream_gpt(c_code):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(c_code), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```c\n','').replace('```','')
        
def stream_claude(c_code):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message,
        messages=[{"role": "user", "content": user_prompt_for(c_code)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace('```cpp\n','').replace('```','')
            
            
def secure(c_code, model):
    if model=="GPT":
        result = stream_gpt(c_code)
    elif model=="Claude":
        result = stream_claude(c_code)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far        
        
       
def execute_python(code):
    try:
        output = io.StringIO()
        sys.stdout = output
        exec(code)
    finally:
        sys.stdout = sys.__stdout__
    return output.getvalue()


def execute_cpp(code):
        write_output(code)
        try:
            compile_cmd = ["g++", "-o", "optimized", "optimized.cpp"]
            compile_result = subprocess.run(compile_cmd, check=True, text=True, capture_output=True)
            run_cmd = ["./optimized"]
            run_result = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
            return run_result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred:\n{e.stderr}"           
sample1 = """
#include <stdio.h>
#include <string.h>

int main() {
    char source[] = "Hello, world!";
    char destination[50];
    scanf("%s", source);
    strcpy(destination, source);  // Copying the string

    printf("Source: %s\n", source);
    printf("Destination: %s\n", destination);

    return 0;
}
"""

#GUI with Gradio 

css = """
.insecure {background-color: #306998;}
.secure {background-color: #050;}
"""

with gr.Blocks(css=css) as ui:
    gr.Markdown("Your C/C++ Code Security Assistant")
    with gr.Row():
        c_code = gr.Textbox(label="Your C/C++ code:", value=sample1, lines=10)
        cpp = gr.Textbox(label="Secure Code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
    with gr.Row():
        convert = gr.Button("Convert code")
    with gr.Row():
        python_run = gr.Button("Run Your Code")
        cpp_run = gr.Button("Run Secure Code")
    with gr.Row():
        python_out = gr.TextArea(label="Result:", elem_classes=["insecure"])
        cpp_out = gr.TextArea(label="Result:", elem_classes=["secure"])

    convert.click(secure, inputs=[c_code, model], outputs=[cpp])
    python_run.click(execute_cpp, inputs=[c_code], outputs=[python_out])
    cpp_run.click(execute_cpp, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)