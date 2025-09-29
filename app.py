# Importing libraries 
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
import requests
from io import BytesIO
import gradio as gr


# API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("OpenAI API Key loaded successfully!")
else:
    raise ValueError("OpenAI API key not found in environment variables.")

# Initializing OpenAI client
client = OpenAI(api_key=openai_api_key)

# Defining State
class State(TypedDict, total=False):
    query: str
    image: str 
    category: str
    sentiment: str
    response: str

# Node functions

def categorize(state: State) -> State:
    """Classify KSG queries into categories"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"Categorize the following Kenya School of Government (KSG) customer query "
                     f"into one of these categories: Admissions, Training, Certificates, General. "
                     f"Query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"Categorize the following Kenya School of Government (KSG) customer query " \
                  f"into one of these categories: Admissions, Training, Certificates, General. " \
                  f"Query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"category": response.choices[0].message.content.strip()}


def analyze_sentiment(state: State) -> State:
    """Analyze sentiment"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"Analyze the sentiment of the following KSG customer query. "
                     f"Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"Analyze the sentiment of the following KSG customer query. " \
                  f"Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"sentiment": response.choices[0].message.content.strip()}


def handle_admissions(state: State) -> State:
    """Admissions/Application support"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"You are a Kenya School of Government support assistant. "
                     f"Provide an admissions/application support response to the following query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"You are a Kenya School of Government support assistant. " \
                  f"Provide an admissions/application support response to the following query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}


def handle_training(state: State) -> State:
    """Training/Program info"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"You are a Kenya School of Government support assistant. "
                     f"Provide a training/program information response to the following query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"You are a Kenya School of Government support assistant. " \
                  f"Provide a training/program information response to the following query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}


def handle_certificates(state: State) -> State:
    """Certificates/Verification support"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"You are a Kenya School of Government support assistant. "
                     f"Provide a certificates/verification support response to the following query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"You are a Kenya School of Government support assistant. " \
                  f"Provide a certificates/verification support response to the following query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}


def handle_general(state: State) -> State:
    """General support response"""
    if state.get("image"):
        content = [
            {"type": "text",
             "text": f"You are a Kenya School of Government support assistant. "
                     f"Provide a general support response to the following query: {state['query']}"},
            {"type": "image_url", "image_url": {"url": state["image"]}}
        ]
    else:
        content = f"You are a Kenya School of Government support assistant. " \
                  f"Provide a general support response to the following query: {state['query']}"

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}
# --- Escalate node ---
def escalate(state: State) -> State:
    """
    Handles escalation when sentiment is negative.
    """
    return {
        "response": "This query has been escalated to a human KSG agent due to its negative sentiment."
    }


# Router
def route_query(state: State) -> str:
    if state.get('sentiment', '').strip().lower() == 'negative':
        return "escalate"
    
    category = state.get('category', '').strip().lower()
    if "admission" in category:
        return "handle_admissions"
    elif "training" in category or "program" in category:
        return "handle_training"
    elif "certificate" in category or "verification" in category:
        return "handle_certificates"
    else:
        return "handle_general"


# workflow
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_admissions", handle_admissions)
workflow.add_node("handle_training", handle_training)
workflow.add_node("handle_certificates", handle_certificates)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_admissions": "handle_admissions",
        "handle_training": "handle_training",
        "handle_certificates": "handle_certificates",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)

workflow.add_edge("handle_admissions", END)
workflow.add_edge("handle_training", END)
workflow.add_edge("handle_certificates", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")
app = workflow.compile()


# Running KSG Support

def url_to_base64(image_url: str) -> str:
    """Download an image from a URL and return a base64 data URI."""
    resp = requests.get(image_url)
    resp.raise_for_status()
    image_bytes = resp.content
    
    mime = "image/jpeg"
    if image_url.lower().endswith(".png"):
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

def run_customer_support(query: str, image: str = None):
    """
    Wrapper to invoke the KSG support workflow.
    `image` can be:
      - None
      - a local file path (string)
      - a base64 data URI (string)
      - a URL to an image (string)
    """
    state = {"query": query}

    if image:
        
        if isinstance(image, str) and image.startswith("http"):
            try:
                image_b64 = url_to_base64(image)
                state["image"] = image_b64
            except Exception as e:
                print("Could not download/convert image from URL:", e)
        
        elif isinstance(image, str) and not image.startswith("data:"):
            with open(image, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            state["image"] = f"data:image/jpeg;base64,{image_b64}"
        else:
            
            state["image"] = image

    # Calling your graph app
    result = app.invoke(state)

    return {
        "category": result.get("category"),
        "sentiment": result.get("sentiment"),
        "response": result.get("response")
    }



def customer_support_ui(query, file_image, url_image):
    image_url = None

    # when file is uploaded
    if file_image is not None:
        with open(file_image.name, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_b64}"
    
    # If URL is provided
    elif url_image:
        image_url = url_image.strip() 

    # Calling backend
    result = run_customer_support(query, image=image_url)
    return result['category'], result['sentiment'], result['response']

iface = gr.Interface(
    fn=customer_support_ui,
    inputs=[
        gr.Textbox(label="Enter your query", placeholder="Type your KSG question here..."),
        gr.File(label="Upload a certificate (optional)"),
        gr.Textbox(label="Or paste an image URL (optional)")
    ],
    outputs=[
        gr.Label(label="Category"),
        gr.Label(label="Sentiment"),
        gr.Textbox(label="Response", lines=15)
    ],
    title="KSG Customer Support Assistant",
    description="Enter a query and optionally upload a certificate or paste an image URL."
)

iface.launch(share=True)