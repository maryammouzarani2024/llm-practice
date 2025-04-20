# This is a very simple usage of a pretrained model in hugging face, you do not need a hugging face token to run it.
# We do not necessarily need gpu to run the model.


from transformers import pipeline

# the transformer librariy A high-level NLP and AI toolkit that gives you easy access to pre-trained models like BERT, GPT, RoBERTa, etc., for tasks like:
# Sentiment analysis, Text generation, Translation, Question answering, Summarization, Image classification (recently!)

# Sentiment Analysis
# Load the sentiment analysis pipeline
#pipeline is responsible to load a pre-trained model , load the tokenizer (to turn text into numbers), apply the model to your input, return interpreted results (e.g., "positive" or "negative", or generated text)


#classifier = pipeline("sentiment-analysis") #when you do not specify a model, it choses a random one
classifier=pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Run it on your input text
result = classifier("I'm super excited to be on the way to LLM mastery!")

# Print the result
print("Sentiment Analysis: ")
print(result)


# Named Entity Recognition

ner = pipeline("ner", grouped_entities=True)
result = ner("Barack Obama was the 44th president of the United States.")
print("Named Entity Recognition:")
print(result)

# Question Answering with Context

question_answerer = pipeline("question-answering")
result = question_answerer(question="Who was the 44th president of the United States?", context="Barack Obama was the 44th president of the United States.")
print("Question Answering with Context")
print(result)


# Text Summarization

summarizer = pipeline("summarization")
text = """The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
It's an extremely popular library that's widely used by the open-source data science community.
It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print("Text Summarization")
print(summary[0]['summary_text'])


# Translation

translator = pipeline("translation_en_to_fr")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print("Translation")
print(result[0]['translation_text'])


# Another translation, showing a model being specified
# All translation models are here: https://huggingface.co/models?pipeline_tag=translation&sort=trending

translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device="cuda")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print("Another translation:")
print(result[0]['translation_text'])


# Text Generation

generator = pipeline("text-generation")
result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
print("Text generation:")
print(result[0]['generated_text'])


# Image Generation
from diffusers import DiffusionPipeline
import torch
image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
    )

text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]
image.show()

