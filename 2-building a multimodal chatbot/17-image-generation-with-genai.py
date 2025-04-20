import base64
from io import BytesIO
from PIL import Image
import openai

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access openai api key using os.getenv
api_key = os.getenv('OPENAI_API_KEY')


# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
    
    
def artist(city):
    image_response=openai.images.generate(
        model="dall-e-3",
        prompt=f"an image representing a vacation in {city} and its touritic areas.",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    
    image_base64=image_response.data[0].b64_json
    image_data=base64.b64decode(image_base64) #decode image into bytes
    return Image.open(BytesIO(image_data))


image=artist("New York")
image.show()