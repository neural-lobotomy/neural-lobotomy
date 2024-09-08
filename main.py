import os

import gradio as gr
from openai import OpenAI

client = OpenAI()

# Function to handle the user input and perform the query
def perform_query(word):
    # Prompt for the OpenAI model
    prompt = f"Please provide a few sentences that include the word or phrase '{word}'."

    try:
        # OpenAI API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant. generate some random sentences that contain the given phrase or concept. dont number the list. give one sentence per line. dont use extra punctuation for each line"},
                      {"role": "user", "content": word}],
            max_tokens=100,
            temperature=0.7
        )

        sentencelist = response.choices[0].message.content
        return sentencelist

    except Exception as e:
        return f"An error occurred: {e}"


# Create a Gradio interface
interface = gr.Interface(
    fn=perform_query,                   # The function to call when the user submits input
    inputs=gr.Textbox(label="Enter a word"),   # Input is a text box
    outputs="text",                     # Output will be shown as text
    title="Word Query",                 # Title of the interface
    description="Enter a word to query for it.", # Description shown to the user
)

interface2 = gr.Interface(
    fn=perform_query,                   # The function to call when the user submits input
    inputs=gr.Textbox(label="Enter a word"),   # Input is a text box
    outputs="text",                     # Output will be shown as text
    title="Word Query",                 # Title of the interface
    description="Enter a word to query for it.", # Description shown to the user
)

# Launch the interface
interface.launch()