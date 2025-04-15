# This is a sample for quantizing a pretrained model to reduce its memory usage. 
# It needs GPU because BitsAndBytesConfig is built for GPU usage only — specifically, NVIDIA GPUs via CUDA
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch


import os
from dotenv import load_dotenv

load_dotenv(override=True)

hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')


login(hugging_face_token)


# instruct models

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI3 = "microsoft/Phi-3-mini-4k-instruct"
GEMMA2 = "google/gemma-2-2b-it"
QWEN2 = "Qwen/Qwen2-7B-Instruct" # exercise for you
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1" # If this doesn't fit it your GPU memory, try others from the hub


messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]


# Quantization Config - this allows us to load the model into memory and use less memory
#Load in 4-bit using the high-quality nf4 format, use bfloat16 for internal math, and apply an extra layer of quantization for better memory use
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, #quantize the model weights to 4-bit precision, 4-bit models are lighter and use less VRAM, which makes them run on smaller GPUs or faster on large ones
    bnb_4bit_use_double_quant=True,# Double quantization, adds a second layer of quantization to the quantization constants themselves, It can help reduce memory even further without much loss in accuracy
    bnb_4bit_compute_dtype=torch.bfloat16,# sets the computation precision for the quantized model to bfloat16 (Brain Float 16 is often used on newer hardware (like NVIDIA A100, H100, etc.) and balances performance with precision better than standard float16)
    bnb_4bit_quant_type="nf4"#Normal Float 4, a specific type of 4-bit quantization developed to retain more information than regular 4-bit formats
)

# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")



model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
print(model)

memory = model.get_memory_footprint() / 1e6
print(f"Memory footprint: {memory:,.1f} MB")


outputs = model.generate(inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0]))


def generate(model, messages):
  tokenizer = AutoTokenizer.from_pretrained(model)
  tokenizer.pad_token = tokenizer.eos_token
  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
  streamer = TextStreamer(tokenizer)
  model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
  outputs = model.generate(inputs, max_new_tokens=80, streamer=streamer)
  del tokenizer, streamer, model, inputs, outputs
  torch.cuda.empty_cache()
  
  
  
  generate(PHI3, messages)
  
  
  messages = [
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]
generate(GEMMA2, messages)