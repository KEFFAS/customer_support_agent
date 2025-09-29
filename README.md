# Customer Support Assistant

## Overview
The **Customer Support Assistant** is a chatbot designed for the **Keffa School of Government (KSG)**.  
It classifies customer queries, analyzes sentiment, and provides relevant responses.  
The system is built using **Gradio** and **LangGraph**, and it leverages the **OpenAI API** to generate intelligent replies.  
If a query carries negative sentiment, the assistant escalates it to a human agent for further handling.

---

## Features
- **Automatic Categorization** – Classifies queries into **Admissions**, **Training**, **Certificates**, or **General** categories.  
- **Sentiment Analysis** – Detects whether a query is **Positive**, **Neutral**, or **Negative**.  
- **Automated Responses** – Generates context-based answers depending on the category.  
- **Escalation Handling** – Automatically forwards negative queries to a human agent.  
- **Interactive UI** – Provides an intuitive web interface built with **Gradio**.

---

## Technologies Used
- **Python 3.8+**
- **LangGraph** – Workflow management  
- **LangChain Core**  
- **OpenAI API** – LLM-based response generation  
- **Gradio** – User interface  
- **python-dotenv** – Environment variable management  
- **IPython** – Visualization (optional)

---

##  Installation

### Prerequisites
- Python ≥ 3.8  
- A valid **OpenAI API Key**  

### Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/customer_support_assistant.git
   cd customer_support_assistant
2. **Create a Virtual Environment and Activate It:**
   ```bash
   # On Linux / macOS
   python -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell)
   python -m venv venv
   venv\Scripts\activate
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Application:**
   ```bash
   python app.py
##  How It Works

1. **User Submits a Query**  
   The user enters a question (and optionally uploads a certificate image) in the Gradio interface.

2. **Categorization**  
   The system classifies the query into one of the following categories:  
   - Admissions  
   - Training  
   - Certificates  
   - General  

3. **Sentiment Analysis**  
   The system analyzes the sentiment of the query and labels it as:  
   - Positive  
   - Neutral  
   - Negative  

4. **Routing & Response**  
   - If the sentiment is **Negative** → The query is escalated to a human KSG agent.  
   - If the sentiment is **Positive** or **Neutral** → The system automatically provides a response based on the category:  
     - **Admissions** → Admissions/application support  
     - **Training** → Training/program information  
     - **Certificates** → Certificates/verification support  
     - **General** → General support response  

5. **Response Display**  
   The category, sentiment, and the generated response are displayed back in the Gradio interface.

