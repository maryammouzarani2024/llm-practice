

import gradio as gr 

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

#to call gradio pass the function name and define the input and output source:
#share=true, means that gradio builds a public url for this page
gr.Interface(fn=shout,  inputs=[gr.Textbox(label="Your message:", lines=6)],
    outputs=[gr.Textbox(label="Response:", lines=8)], flagging_mode="never").launch(share=True)