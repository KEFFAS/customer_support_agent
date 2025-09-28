%pip install gradio

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
    messages = [{"role": "user", "content": f"Categorize the following Kenya School of Government (KSG) customer query "
                                           f"into one of these categories: Admissions, Training, Certificates, General. "
                                           f"Query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"category": response.choices[0].message.content.strip()}

def analyze_sentiment(state: State) -> State:
    messages = [{"role": "user", "content": f"Analyze the sentiment of the following KSG customer query. "
                                           f"Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"sentiment": response.choices[0].message.content.strip()}

def handle_admissions(state: State) -> State:
    messages = [{"role": "user", "content": f"You are a Kenya School of Government support assistant. "
                                           f"Provide an admissions/application support response to the following query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}

def handle_training(state: State) -> State:
    messages = [{"role": "user", "content": f"You are a Kenya School of Government support assistant. "
                                           f"Provide a training/program information response to the following query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}

def handle_certificates(state: State) -> State:
    messages = [{"role": "user", "content": f"You are a Kenya School of Government support assistant. "
                                           f"Provide a certificates/verification support response to the following query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}

def handle_general(state: State) -> State:
    messages = [{"role": "user", "content": f"You are a Kenya School of Government support assistant. "
                                           f"Provide a general support response to the following query: {state['query']}"}]
    if state.get("image"):
        messages.append({"role": "user", "content": {"type": "image_url", "image_url": state["image"]}})
    
    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages,
        temperature=1
    )
    return {"response": response.choices[0].message.content.strip()}

def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human KSG agent due to its negative sentiment."}

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
def run_customer_support(query: str, image: str = None) -> dict:
    """
    Handles text and optional image, returning category, sentiment, and response.
    """
    state = {"query": query}
    if image:
        state["image"] = image

    result = app.invoke(state)
    return {
        "category": result.get("category"),
        "sentiment": result.get("sentiment"),
        "response": result.get("response")
    }


# Wrapper Gradio UI
def customer_support_ui(query, image):
    image_url = None
    if image is not None:
        
        with open(image.name, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_b64}"  # 
    
    # Calling existing workflow
    result = run_customer_support(query, image=image_url)
    return result['category'], result['sentiment'], result['response']

# Gradio interface
iface = gr.Interface(
    fn=customer_support_ui,
    inputs=[
        gr.Textbox(label="Enter your query", placeholder="Type your KSG question here..."),
        gr.File(label="Upload a certificate (optional)")
    ],
    outputs=[
        gr.Textbox(label="Category"),
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Response",lines=15)
    ],
    title="Ksg Customer Support Assistant",
    description="Enter a query and optionally upload a certificate. The assistant will return category, sentiment, and a response."
)

# Launching the interface
iface.launch()