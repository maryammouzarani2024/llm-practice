#Here we want to compare the tokenizers of various models:
#Since meta llama model requires authentication, we have to login first to hugging face and get a token

from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv(override=True)

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')


login(hugging_face_token)


from transformers import AutoTokenizer


PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL_NAME = "bigcode/starcoder2-3b"




tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"
print("Let's compare how tokenizers of different models tokenize a sample text.")
print("the text is: "+ text)

print("llama3.1 model: ")
tokens = tokenizer.encode(text)
print(tokens)
print("there are " +str(len(tokens))+ "tokens in this sentence.")
print("original text:  "+ tokenizer.decode(tokens))
print("The list of tokens: ")
print(tokenizer.batch_decode(tokens))
print("The vocabulary of the tokenizer for Meta-Llama-3.1 model:")
print(tokenizer.vocab)

print("special tokens, e.g. begin of text, end of text, for Meta-llama-3.1 model: ")
print(tokenizer.get_added_vocab())

print("qwen model: ")

qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL_NAME)
print(qwen2_tokenizer.encode(text))



print("phi model: ")
phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME)

print(phi3_tokenizer.encode(text))



print("comparing the instruct variants of models tokenization methods: ")



messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
print("llama chat tokenizer: ")
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', trust_remote_code=True)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

print("phi chat tokenizer: ")
print(phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

print("Qwen chat tokenizer: ")
print(qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


print("Tokenizing the codes: ")
starcoder2_tokenizer = AutoTokenizer.from_pretrained(STARCODER2_MODEL_NAME, trust_remote_code=True)
code = """
def hello_world(person):
  print("Hello", person)
"""
tokens = starcoder2_tokenizer.encode(code)
for token in tokens:
  print(f"{token}={starcoder2_tokenizer.decode(token)}")