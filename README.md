# TITLE: Streamlit Application for PDF Interaction Using Language Models

The codes in the provided excerpts are related to creating a Streamlit application that allows users to upload PDF files and interact with them using a language model (LLM). The application includes functionalities for displaying PDFs, adding them to a knowledge base, setting up a chat interface, and saving/loading chat history.

## Table of Contents

1. [Import Libraries](#import-libraries)
2. [Configure Embedchain App](#configure-embedchain-app)
3. [Display PDF Function](#display-pdf-function)
4. [Streamlit Title and Preparation](#streamlit-title-and-preparation)
5. [Sidebar for PDF Upload](#sidebar-for-pdf-upload)
6. [Add PDF to Knowledge Base](#add-pdf-to-knowledge-base)
7. [Set Chat Interface](#set-chat-interface)
8. [Save Chat History Function](#save-chat-history-function)
9. [Load Chat History Function](#load-chat-history-function)
10. [Hash Files Function](#hash-files-function)
11. [Logging Chat Discussion](#logging-chat-discussion)
12. [Save and Load Chat History](#save-and-load-chat-history)
13. [Future-proof Query Method](#future-proof-query-method)

## Import Libraries
```python
import os
import tempfile
from click import prompt
from httpx import stream
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import messages
```

## Configure Embedchain App
```python
def embedchainbot(db_path):
    return App.from_config(
      config = {
        "llm": {"provider": "ollama", "config": {"model": "3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, stream: True, "base_url": "https://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "3.2:3b", "base_url": "https://localhost:11434"}},
      }
    )
```

## Display PDF Function
```python
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)
```

## Streamlit Title and Preparation
```python
st.title("Chat with your PDF's using LLM's")
st.caption("This App lets you chat with your model")

db_path = tempfile.mkdtemp() # db to start pdf temporarily

if 'app' not in st.session_state:
    st.session_state.app = embedchainbot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []
```

## Sidebar for PDF Upload
```python
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)
```

## Add PDF to Knowledge Base
```python
if st.button("Submit PDF"):
    with st.spinner("Adding PDF to the knowledge base..."):
       with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
           temp.write(pdf_file.getvalue())
           st.session_state.app.add(temp.name, data_type="pdf_file")
       os.remove(temp.name)
    st.success(f"Added {pdf_file.name} to the knowledge base!")
```

## Set Chat Interface
```python
for i,msg in enumerate(st.session_state.messages):
    message(msg["Content"], is_user=msg["role"] == "user",key=str(i))
    
if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "Content": prompt})
    message(prompt, is_user=True)

# User query and display response
with st.spinner("Thinking..."):
    response = st.session_state.app.chat(prompt)
    st.session_state.messages.append({"role": "assistant", "Content": response})
    message(response)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")  
```

## Save Chat History Function
```python
def save_chat_history(chat_history, file_path):
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)
```

## Load Chat History Function
```python
def load_chat_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []
```

## Hash Files Function
```python
def hash_files(files):
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)  # Reset file pointer to the beginning
        hasher.update(file.read())
    return hasher.hexdigest()
```

## Logging Chat Discussion
```python
import os
import tempfile
import base64
import streamlit as st
from click import prompt
from httpx import stream
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger

# Set page configuration
st.set_page_config("Chat with your PDF's using LLM's")

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

vector_db = 'D:\Ollama\vectordb'

# configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    return App.from_config(
       config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
      }
    )

def embedchainbot(db_path):
    # Initialize OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3.2:3b", base_url="http://localhost:11434")
    
    # Clear existing Chroma instance if it exists
    if is_file_in_use(os.path.join(db_path, 'chroma.sqlite3')):
        logger.error(f"File {os.path.join(db_path, 'chroma.sqlite3')} is in use by another process.")
        st.error(f"File {os.path.join(db_path, 'chroma.sqlite3')} is in use by another process.")
        return None
        
    # Initialize App
    try:
        app = App.from_config(
           config = {
            "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                      "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}}
          }
        )
        app.embeddings = embeddings  # Set the embeddings instance
        st.session_state.app = app  # Store the app instance in session state
    except ValueError as e:
        logger.error(f"Error initializing Chroma: {e}")
        st.error(f"Error initializing Chroma: {e}")
        return None
    
    return app

# Configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    # Initialize OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="llama3.2:3b", base_url="http://localhost:11434")
    config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                 "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
    }
    app = App.from_config(config)
    app.embeddings = embeddings  # Set the embeddings instance
    return app

# Add function to display PDF's in the streamlit app
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit title and preparation for the chat
st.title("Chat with your PDF's using LLM's")
st.caption("This App lets you chat with your model")

db_path = tempfile.mkdtemp() # db to start pdf temporarily

if 'app' not in st.session_state:
    st.session_state.app = embedchainbot(db_path)
if 'message' not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload 
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

# Add PDF to knowledge base
if st.button("Submit PDF"):
    if pdf_file:
        with st.spinner("Adding PDF to the knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(pdf_file.getvalue())
                temp.flush()  # Ensure data is written to the file
                st.session_state.app.add(temp.name, data_type="pdf_file")
            os.remove(temp.name)
        st.success(f"Added {pdf_file.name} to the knowledge base!")
    else:
        st.error("Please upload a PDF file before submitting.")

# Set Chat Interface
for i, msg in enumerate(st.session_state.messages):
    message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "Content": prompt})
    message(prompt, is_user=True)
    logger.info(f"User: {prompt}")

# User query and display response
with st.spinner("Thinking..."):
    response = st.session_state.app.chat(prompt)
    st.session_state.messages.append({"role": "assistant", "Content": response})
    message(response)
    logger.info(f"Chatbot: {response}")

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")
    logger.info("Chat cleared!")
```

## Save and Load Chat History
```python
import os
import tempfile
import base64
import hashlib
import json
import streamlit as st
from click import prompt
from httpx import stream
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger

# Set page configuration
st.set_page_config("Chat with your PDF's using LLM's")

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO) 

# configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    return App.from_config(
       config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
      }
    )

# Add function to display PDF's in the streamlit app
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)

# Function to save chat history
def save_chat_history(chat_history, file_path):
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)

# Function to load chat history
def load_chat_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save in Hash_file
def hash_files(files):
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)  # Reset file pointer to the beginning
        hasher.update(file.read())
    return hasher.hexdigest()

# Streamlit title and preparation for the chat
st.title("Chat with your PDF's using LLM's")
st.caption("This App lets you chat with your model")

db_path = tempfile.mkdtemp() # db to start pdf temporarily

if 'app' not in st.session_state:
    st.session_state.app = embedchainbot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load chat history if available
chat_history_file = os.path.join(log_dir, 'chat_history.json')
st.session_state.messages = load_chat_history(chat_history_file)

# Sidebar for PDF upload 
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

# Add PDF to knowledge base
if st.button("Submit PDF"):
    if pdf_file:
        with st.spinner("Adding PDF to the knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(pdf_file.getvalue())
                temp.flush()  # Ensure data is written to the file
                st.session_state.app.add(temp.name, data_type="pdf_file")
            os.remove(temp.name)
        st.success(f"Added {pdf_file.name} to the knowledge base!")
    else:
        st.error("Please upload a PDF file before submitting.")

# Set Chat Interface
for i, msg in enumerate(st.session_state.messages):
    message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "Content": prompt})
    message(prompt, is_user=True)
    logger.info(f"User: {prompt}")

    # Save chat history
    save_chat_history(st.session_state.messages, chat_history_file)

# User query and display response
with st.spinner("Thinking..."):
    response = st.session_state.app.chat(prompt)
    st.session_state.messages.append({"role": "assistant", "Content": response})
    message(response)
    logger.info(f"Chatbot: {response}")

    # Save chat history
    save_chat_history(st.session_state.messages, chat_history_file)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")
    logger.info("Chat cleared!")

    # Save chat history
    save_chat_history(st.session_state.messages, chat_history_file)
```

## Future-proof Query Method
```python
import os
import tempfile
import base64
import hashlib
import json
import streamlit as st
from click import prompt
from httpx import stream
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger

# Set page configuration
st.set_page_config("Chat with your PDF's using LLM's")

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO) 

# configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    return App.from_config(
       config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
      }
    )

# Add function to display PDF's in the streamlit app
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)

# Function to save chat history
def save_chat_history(chat_history, dir, file_name):
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)

# Function to load chat history
def load_chat_history(dir, file_name):
    file_path = os.path.join(dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save in Hash_file
def hash_files(files):
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)  # Reset file pointer to the beginning
        hasher.update(file.read())
    return hasher.hexdigest()

# Streamlit title and preparation for the chat
st.title("Chat with your PDF's using LLM's")
st.caption("This App lets you chat with your model")

db_path = tempfile.mkdtemp() # db to start pdf temporarily

if 'app' not in st.session_state:
    st.session_state.app = embedchainbot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load chat history if available
chat_history_file = 'chat_history.json'
st.session_state.messages = load_chat_history(log_dir, chat_history_file)

# Sidebar for PDF upload 
with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        st.subheader("PDF Preview")
        display_pdf(pdf_file)

# Add PDF to knowledge base
if st.button("Submit PDF"):
    if pdf_file:
        with st.spinner("Adding PDF to the knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                temp.write(pdf_file.getvalue())
                temp.flush()  # Ensure data is written to the file
                st.session_state.app.add(temp.name, data_type="pdf_file")
            os.remove(temp.name)
        st.success(f"Added {pdf_file.name} to the knowledge base!")
    else:
        st.error("Please upload a PDF file before submitting.")

# Set Chat Interface
for i, msg in enumerate(st.session_state.messages):
    message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "Content": prompt})
    message(prompt, is_user=True)
    logger.info(f"User: {prompt}")

    # Save chat history
    save_chat_history(st.session_state.messages, log_dir, chat_history_file)

# User query and display response
with st.spinner("Thinking..."):
    response = st.session_state.app.chat(prompt)
    if isinstance(response, tuple):
        response = response[0]  # Extract the answer from the tuple
    st.session_state.messages.append({"role": "assistant", "Content": response})
    message(response)
    logger.info(f"Chatbot: {response}")

    # Save chat history
    save_chat_history(st.session_state.messages, log_dir, chat_history_file)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared!")
    logger.info("Chat cleared!")

    # Save chat history
    save_chat_history(st.session_state.messages, log_dir, chat_history_file)


import os
import tempfile
import base64
import hashlib
import json
import streamlit as st
from click import prompt
from httpx import stream
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger

# Set page configuration
st.set_page_config("Chat with your PDF's using LLM's")

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO) 

# configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    return App.from_config(
       config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
      }
    )

# Add function to display PDF's in the streamlit app
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)

# Function to save chat history
def save_chat_history(chat_history, dir, file_name):
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)

# Function to load chat history
def load_chat_history(dir, file_name):
    file_path = os.path.join(dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save in Hash_file
def hash_files(files):
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)  # Reset file pointer to the beginning
        hasher.update(file.read())
    return hasher.hexdigest()
  
def main():
    # Streamlit title and preparation for the chat
    st.title("Chat with your PDF's using LLM's")
    st.caption("This App lets you chat with your model")

    db_path = tempfile.mkdtemp()  # db to start pdf temporarily

    if 'app' not in st.session_state:
        st.session_state.app = embedchainbot(db_path)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load chat history if available
    chat_history_file = 'chat_history.json'
    st.session_state.messages = load_chat_history(log_dir, chat_history_file)

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file:
            st.subheader("PDF Preview")
            display_pdf(pdf_file)

    # Add PDF to knowledge base
    if st.button("Submit PDF"):
        if pdf_file:
            with st.spinner("Adding PDF to the knowledge base..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(pdf_file.getvalue())
                    temp.flush()  # Ensure data is written to the file
                    st.session_state.app.add(temp.name, data_type="pdf_file")
                os.remove(temp.name)
            st.success(f"Added {pdf_file.name} to the knowledge base!")
        else:
            st.error("Please upload a PDF file before submitting.")

    # Set Chat Interface
    for i, msg in enumerate(st.session_state.messages):
        message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

    if prompt := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append({"role": "user", "Content": prompt})
        message(prompt, is_user=True)
        logger.info(f"User: {prompt}")

        # Save chat history
        save_chat_history(st.session_state.messages, log_dir, chat_history_file)

        # User query and display response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.app.chat(prompt)
                if isinstance(response, tuple):
                    response = response[0]  # Extract the answer from the tuple
                st.session_state.messages.append({"role": "assistant", "Content": response})
                message(response)
                logger.info(f"Chatbot: {response}")
            except ValueError as e:
                st.error(f"Error: {e}")
                logger.error(f"Error during chat: {e}")

            # Save chat history
            save_chat_history(st.session_state.messages, log_dir, chat_history_file)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")
        logger.info("Chat cleared!")

        # Save chat history
        save_chat_history(st.session_state.messages, log_dir, chat_history_file)

if __name__ == "__main__":
    main()

```
The error message indicates that the directory specified by data_dir does not exist. To resolve this, you need to ensure that the directory exists before attempting to load documents from it. You can add a check to create the directory if it does not exist. Here is the updated code:

```python
import os
import tempfile
import base64
import hashlib
import json
import streamlit as st
import argparse
import shutil
from click import prompt
from httpx import stream
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger
from chromadb.api.types import Documents
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Set page configuration
st.set_page_config("Chat with your PDF's using LLM's")

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

vectordb_dir = r"D:\Ollama\vectordb"  # Directory to store the Chroma database
    if not os.path.exists(vectordb_dir):
        os.makedirs(vectordb_dir)
        
# configure embedchain App - we are using Ollama specifically Llama3.2:3b
def embedchainbot(db_path):
    return App.from_config(
       config = {
        "llm": {"provider": "ollama", "config": {"model": "llama3.2:3b", "max_tokens": 250,
                                                  "temperature": 0.5, "stream": True, "base_url": "http://localhost:11434"}},
        "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
        "embedder": {"provider": "ollama", "config": {"model": "llama3.2:3b", "base_url": "http://localhost:11434"}},
      }
    )

# Add function to display PDF's in the streamlit app
def display_pdf(file):
  base64_pdf = base64.b64encode(file.read()).decode('utf-8')
  pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
  st.markdown(pdf_display, unsafe_allow_html=True)

# Function to save chat history
def save_chat_history(chat_history, dir, file_name):
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(chat_history, f)

# Function to load chat history
def load_chat_history(dir, file_name):
    file_path = os.path.join(dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

# Save in Hash_file
def hash_files(files):
    hasher = hashlib.md5()
    for file in files:
        file.seek(0)  # Reset file pointer to the beginning
        hasher.update(file.read())
    return hasher.hexdigest()

# Function to load documents from a directory
def load_documents(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# Function to split documents into chunks
def split_documents(documents: list[Documents]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Function to calculate unique chunk IDs
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# Function to initialize Chroma
def initialize_chroma(db_path, collection_name="default_collection"):
    db = Chroma(
        persist_directory=db_path,
        embedding_function=get_embedding_function(),
        collection_name=collection_name
    )
    return db

# Function to add chunks to Chroma database
def add_to_chroma(chunks: list[Documents], db_path, collection_name="default_collection"):
    try:
        db = initialize_chroma(db_path, collection_name)
        chunks_with_ids = calculate_chunk_ids(chunks)
        logger.info(f"Processing {len(chunks_with_ids)} chunks")
        for chunk in chunks_with_ids:
            logger.info(f"Chunk ID: {chunk.metadata['id']}, Source: {chunk.metadata.get('source')}, Page: {chunk.metadata.get('page')}")

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            logger.info(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            logger.info("✅ No new documents to add")
    except ValueError as e:
        logger.error(f"Error initializing Chroma: {e}")
        st.error(f"Error initializing Chroma: {e}")
        clear_chroma_instance(db_path)
        db = initialize_chroma(db_path, collection_name)
        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
        if new_chunks:
            db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
            db.persist()

# Function to clear the database
def clear_database(db_path):
    if os.path.exists(db_path):
        try:
            client = chromadb.Client(chromadb.config.Settings(persist_directory=db_path))
            client.reset()
        except Exception as e:
            logger.error(f"Error closing Chroma instance: {e}")
        shutil.rmtree(db_path)
        os.makedirs(db_path)

def main():
    vectordb_dir = r"D:\Ollama\vectordb"  # Directory to store the Chroma database
    if not os.path.exists(vectordb_dir):
        os.makedirs(vectordb_dir)

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database(vectordb_dir)

    # Create (or update) the data store.
    data_dir = 'data'  # Directory containing PDF files
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    add_to_chroma(chunks, vectordb_dir)

    # Streamlit title and preparation for the chat
    st.title("Chat with your PDF's using LLM's")
    st.caption("This App lets you chat with your model")

    db_path = tempfile.mkdtemp()  # db to start pdf temporarily

    if 'app' not in st.session_state:
        st.session_state.app = embedchainbot(db_path)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load chat history if available
    chat_history_file = 'chat_history.json'
    st.session_state.messages = load_chat_history(log_dir, chat_history_file)

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file:
            st.subheader("PDF Preview")
            display_pdf(pdf_file)

    # Add PDF to knowledge base
    if st.button("Submit PDF"):
        if pdf_file:
            with st.spinner("Adding PDF to the knowledge base..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(pdf_file.getvalue())
                    temp.flush()  # Ensure data is written to the file
                    st.session_state.app.add(temp.name, data_type="pdf_file")
                os.remove(temp.name)
            st.success(f"Added {pdf_file.name} to the knowledge base!")
        else:
            st.error("Please upload a PDF file before submitting.")

    # Set Chat Interface
    for i, msg in enumerate(st.session_state.messages):
        message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

    if prompt := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append({"role": "user", "Content": prompt})
        message

(prompt

, is_user=True)
        logger.info(f"User: {prompt}")

        # Save chat history
        save_chat_history(st.session_state.messages, log_dir, chat_history_file)

        # User query and display response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.app.chat(prompt)
                if isinstance(response, tuple):
                    response = response[0]  # Extract the answer from the tuple
                st.session_state.messages.append({"role": "assistant", "Content": response})
                message(response)
                logger.info(f"Chatbot: {response}")
            except ValueError as e:
                st.error(f"Error: {e}")
                logger.error(f"Error during chat: {e}")

            # Save chat history
            save_chat_history(st.session_state.messages, log_dir, chat_history_file)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared!")
        logger.info("Chat cleared!")

        # Save chat history
        save_chat_history(st.session_state.messages, log_dir, chat_history_file)

if __name__ == "__main__":
    main()
```

In this updated code, the load_documents function now checks if the data_dir exists and creates it if it does not. This should resolve the `FileNotFoundError` and allow the script to run correctly.

The `data_dir` is a directory where your PDF files are stored. When you run the script, it processes these PDF files, splits them into chunks, and then stores these chunks in the Chroma database.

Here's a step-by-step explanation of what happens:

1. **PDF Files in `data_dir`**:
   - The `data_dir` contains PDF files that you want to process and store in the Chroma database.

2. **Loading Documents**:
   - The `load_documents` function iterates over all PDF files in the `data_dir`, loads each PDF using `PyPDFLoader`, and returns a list of documents.

3. **Splitting Documents into Chunks**:
   - The `split_documents` function takes the list of documents and splits each document into smaller chunks using `RecursiveCharacterTextSplitter`. This is done to make the text more manageable and to facilitate better search and retrieval performance.

4. **Calculating Unique Chunk IDs**:
   - The `calculate_chunk_ids` function assigns a unique ID to each chunk based on its source (PDF file) and page number. This helps in uniquely identifying each chunk in the database.

5. **Storing Chunks in Chroma Database**:
   - The `add_to_chroma` function initializes the Chroma database and adds the chunks to it. It checks for existing documents in the database to avoid duplicates and only adds new chunks.

6. **Chroma Database Structure**:
   - The Chroma database stores the chunks of text along with their metadata (such as source, page number, and unique ID). The metadata helps in organizing and retrieving the chunks efficiently.
   - The chunks are stored in a table (or collection) in the Chroma database. Each chunk is stored with fields like `id`, `title`, `content`, `page`, etc.
   - The `embedding_fulltext_search_content` and `c0` are likely related to the full-text search capabilities of the Chroma database. These fields store the embeddings (vector representations) of the text content, which are used for efficient search and retrieval.

Here is a simplified representation of the process:

1. **PDF Files in `data_dir`**:
   ```
   data/
   ├── file1.pdf
   ├── file2.pdf
   └── file3.pdf
   ```

2. **Loading Documents**:
   - Each PDF file is loaded and converted into a document object.

3. **Splitting Documents into Chunks**:
   - Each document is split into smaller chunks of text.

4. **Calculating Unique Chunk IDs**:
   - Each chunk is assigned a unique ID based on its source and page number.

5. **Storing Chunks in Chroma Database**:
   - The chunks are stored in the Chroma database with their metadata.

6. **Chroma Database Structure**:
   - The database stores the chunks with fields like `id`, `title`, `content`, `page`, etc.
   - The embeddings of the text content are stored in fields like `embedding_fulltext_search_content` and `c0` for efficient search and retrieval.

By following this process, the Chroma database can efficiently store, organize, and retrieve chunks of text from your PDF files, enabling you to perform full-text search and other operations on the stored content.
