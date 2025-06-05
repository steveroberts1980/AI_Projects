import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import gradio as gr

# environment
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')

# initialize
# NOTE - option to use ultra-low cost models by uncommenting last 2 lines

openai = OpenAI()
claude = anthropic.Anthropic()
OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# Want to keep costs ultra-low? Uncomment these lines:
# OPENAI_MODEL = "gpt-4o-mini"
# CLAUDE_MODEL = "claude-3-haiku-20240307"
python_snippet = 'print("Hello world!")'
languages = ["Python", "C++", "C#", "Java", "Javascript", "Ruby", "Golang", "Rust"]
languages.sort()

def get_system_message(source_lang, dest_lang):
    system_message = f"You are an assistant that reimplements {source_lang} code in high performance {dest_lang}. "
    system_message += f"Respond only with {dest_lang} code; use comments sparingly and do not provide any explanation other than occasional comments. "
    system_message += "The response needs to produce an identical output in the fastest possible time."
    return system_message

def user_prompt_for(code, source_lang, dest_lang):
    user_prompt = f"Rewrite this {source_lang} code in {dest_lang} with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += f"Respond only with {dest_lang} code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no type overflows. Include all necessary imports/packages.\n\n"
    user_prompt += code
    return user_prompt

def messages_for(code, source_lang, dest_lang):
    return [
        {"role": "system", "content": get_system_message(source_lang, dest_lang)},
        {"role": "user", "content": user_prompt_for(code, source_lang, dest_lang)}
    ]

def stream_gpt(code, source_lang, dest_lang):
    stream = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages_for(code, source_lang, dest_lang),
        stream=True
    )
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace(f'```{dest_lang.lower()}\n','').replace('```','')

def stream_claude(code, source_lang, dest_lang):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=get_system_message(source_lang, dest_lang),
        messages=[{"role": "user", "content": user_prompt_for(code, source_lang, dest_lang)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield reply.replace(f'```{dest_lang.lower()}\n','').replace('```','')

def convert_code(source_code, source_lang, dest_lang, model):
    if model=="GPT":
        result = stream_gpt(source_code, source_lang, dest_lang)
    elif model=="Claude":
        result = stream_claude(source_code, source_lang, dest_lang)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far

def update_label(selected_value, source_code):
    return gr.Textbox(label=f"{selected_value} code", value=source_code, lines=10)

with gr.Blocks() as demo:
    gr.Markdown("## Code Language Converter")
    with gr.Row():
        source_language = gr.Dropdown(languages, label="Select source language", value="Python")
        destination_language = gr.Dropdown(languages, label="Select destination language", value="C++")
    with gr.Row():
        source_code = gr.Textbox(label="Python code", value=python_snippet, lines=10)
        destination_code = gr.Textbox(label="C++ code:", lines=10)
    with gr.Row():
        model = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
    with gr.Row():
        convert = gr.Button("Convert code")

        source_language.change(fn=update_label, inputs=[source_language, source_code], outputs=source_code)
        destination_language.change(fn=update_label, inputs=[destination_language, destination_code], outputs=destination_code)

    convert.click(
        convert_code,
        inputs=[source_code, source_language, destination_language, model],
        outputs=[destination_code]
    )

if __name__ == "__main__":
    demo.launch()
# ui.launch(inbrowser=True)
