import os
from openai import OpenAI

def generate_response(user_input, user_api_key):
    client = OpenAI(
    api_key=user_api_key
    )

    print("STARTED REQUEST: " + str(user_input))
    
    response = client.images.generate(
        model = "dall-e-3",
        prompt= (user_input),
        size = "1024x1024",
        quality="standard",
        n=1
    )
        
    image_url = response.data[0].url

    print (image_url)

    return image_url