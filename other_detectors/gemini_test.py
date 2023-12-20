import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

def multiturn_generate_content():
    config = {
        "max_output_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1
    }
    model = GenerativeModel("gemini-pro")
    chat = model.start_chat()
    res = chat.send_message("""hello""", generation_config=config)

    print(res.__class__)
    print(res)