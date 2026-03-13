## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Text data, especially from the web or internal documents, is vast and unstructured. Manually extracting specific pieces of information, such as names of people (PER), organizations (ORG), locations (LOC), or other miscellaneous entities (MISC), is time-consuming, prone to error, and inefficient.

The problem is to develop an automated system that can process any given text, accurately identify these named entities, and classify them into their predefined categories. This system must also be wrapped in a simple, accessible web interface, allowing non-technical users to easily input text and visualize the model's predictions for testing and validation.

### DESIGN STEPS:
### Step1: Install Required Libraries:
Install the necessary Python libraries such as transformers, torch, and gradio.

### Step2: Load the NER Model:
Use the transformers library to load a fine-tuned BART model for NER, for example, dslim/bart-large-ner.

### Step3: Prepare the Output Format:
Organize the model output so that entities and their labels can be clearly highlighted in the interface.

### Step4: Build the Gradio Interface:
Create a user-friendly interface with a text input box and a highlighted output display using the Gradio framework.

### Step5: Customize the Interface:
Add colors for each entity type (like red for people, blue for organizations, green for locations) and include a title and description.


### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    text = response.content.decode("utf-8").strip()

    # Handle extra data safely
    try:
        # Try parsing as normal JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If response contains multiple JSON objects, take the first valid one
        parts = text.split("\n")
        for part in parts:
            try:
                return json.loads(part)
            except Exception:
                continue
        raise ValueError(f"Invalid JSON returned from model: {text}")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last = merged_tokens[-1]
            last['word'] += token['word'].replace('##', '')
            last['end'] = token['end']
            last['score'] = (last['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not isinstance(output, list):
        raise ValueError(f"Unexpected model output: {output}")
    merged_tokens = merge_tokens(output)
    return {"text": input_text, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find named entities using the dslim/bert-base-NER model via Hugging Face Inference API.",
    allow_flagging="never",
    examples=[
        "My name is mohan, I work at DeepLearningAI and live in Chennai.",
        "mohan lives in Chennai and works at HuggingFace."
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT3", 7860)))
```


### OUTPUT:
<img width="1261" height="685" alt="image" src="https://github.com/user-attachments/assets/5e7a3e65-5357-4a1b-b129-cb3ea7d93bb3" />


### RESULT:
To develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation excuted successfully.
