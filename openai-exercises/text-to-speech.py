# openai-exercises/text_to_speech.py
import openai
import os

# Set your API key as an environment variable before running
# export OPENAI_API_KEY='your-api-key'
openai.api_key = os.getenv("OPENAI_API_KEY")

def text_to_speech_example():
    """Simple example of OpenAI's text-to-speech API"""
    try:
        # Generate speech from text
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input="Hello! This is an example of text-to-speech generation using OpenAI's API."
        )
        
        # Save the audio to a file
        output_file = "speech_output.mp3"
        response.write_to_file(output_file)
        
        print(f"Audio generated and saved as '{output_file}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("OpenAI Text-to-Speech Example")
    print("----------------------------")
    text_to_speech_example()