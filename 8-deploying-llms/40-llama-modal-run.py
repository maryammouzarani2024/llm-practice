import modal

from llama_code import app, generate

with modal.enable_output():
    with app.run():
            result=generate.remote("Hi, how can I improve my English?")
print(result)