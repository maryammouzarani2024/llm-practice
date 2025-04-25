import modal
from hello import app, hello_europe


with app.run():
    reply=hello_europe.local()
print(reply)

with app.run():
    reply=hello_europe.remote()
print(reply)
