import os

import gradio as gr
from openai import OpenAI

from lobotomy import compute_lobotomy_vector_function

client = OpenAI()

# Function to handle the user input and perform the query
def perform_query(word):

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

with gr.Blocks() as demo:
    gr.Markdown("# OpenAI Query Interface using Gradio Blocks")
    gr.Markdown("Enter a word or phrase and get sentences that include it.")

    with gr.Row():
        word_input = gr.Textbox(label="Enter a word or phrase")
    with gr.Row():
        lobotomy_vector = gr.Textbox(label="lobotomy vector")
    with gr.Row():
        lobotomy_text = gr.Textbox(label="Generated Sentences")

    with gr.Row():
        submit_button = gr.Button("Submit")
        compute_lobotomy_vector = gr.Button("compute lobotomy_vector")

    submit_button.click(perform_query, inputs=word_input, outputs=lobotomy_text)
    compute_lobotomy_vector.click(compute_lobotomy_vector_function, inputs=lobotomy_text, outputs=lobotomy_vector)



# Launch the interface
demo.launch()