from groq import Groq
import base64
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("api_key")


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_LLM(image_path):
   

    # Getting the base64 string
    base64_image = encode_image(image_path)

    client = Groq(api_key=key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "give me only the text here in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content 





