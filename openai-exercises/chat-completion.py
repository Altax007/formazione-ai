# openai-exercises/chat_completion.py
import openai
import os

# Set your API key as an environment variable before running
# export OPENAI_API_KEY='your-api-key' oppure $env:OPENAI_API_KEY = "your-api-key"
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_completion_example():
    """Simple example of OpenAI's chat completion API"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain what a vector database is in one sentence."}
            ],
            max_tokens=100
        )
        
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("OpenAI Chat Completion Example")
    print("------------------------------")
    chat_completion_example()