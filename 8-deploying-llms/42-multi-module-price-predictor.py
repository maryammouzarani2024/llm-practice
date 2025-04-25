# imports

import os
import re
import math
import json
from tqdm import tqdm
import random
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from openai import OpenAI


import os
from dotenv import load_dotenv

load_dotenv(override=True)

hugging_face_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
login(hugging_face_token)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()


DB = "products_vectorstore"


with open('train.pkl', 'rb') as file:
    train = pickle.load(file)


#test:
# print(train[10].prompt)

#create a Chroma Datastore
client = chromadb.PersistentClient(path=DB)

# Check if the collection exists and delete it if it does
collection_name = "products"
existing_collection_names = client.list_collections()
if collection_name in existing_collection_names:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection: {collection_name}")

collection = client.create_collection(collection_name)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# test, Pass in a list of texts, get back a numpy array of vectors
# vector = model.encode(["Well hi there"])[0]
# print(vector)

print(train[0].prompt)


def description(item):
    text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
    return text.split("\n\nPrice is $")[0]
print("AFTER")
print(description(train[0]))