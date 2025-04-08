#Here we want to compare the tokenizers of various models:
#Since meta llama model requires authentication, we have to login first to hugging face and get a token

from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv(override=True)

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')


login(hugging_face_token)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)

text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
print(tokens)