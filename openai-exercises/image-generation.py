# openai-exercises/image_generation.py
import openai
import os

# Set your API key as an environment variable before running
# export OPENAI_API_KEY='your-api-key'
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image_example():
    """Simple example of OpenAI's DALL-E image generation API"""
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt="A cute robot reading a book in a library, digital art style",
            size="1024x1024",
            n=1
        )
        
        # Get the generated image URL
        image_url = response.data[0].url
        print("Generated Image URL:", image_url)
        print("You can copy and paste this URL in your browser to view the image")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("OpenAI Image Generation Example")
    print("------------------------------")
    generate_image_example()