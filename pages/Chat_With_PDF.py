import os
import base64
import hashlib
import json
import shutil
import argparse
import time

import streamlit as st
from embedchain import App
from streamlit_chat import message
from functions.logger import Logger
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Updated import
# Updated import
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from langchain.schema.document import Document

from pypdf.errors import PdfReadError, PdfStreamError

def Chat_With_PDF():
  # Ensure the logs directory exists
  log_dir = 'logs'
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)

  # Initialize the Logger instance
  logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

  def get_embedding_function():
      embeddings = OllamaEmbeddings(model="nomic-embed-text")
      return embeddings

  CHROMA_PATH = (r"D:\Ollama\vectordb")
  DATA_PATH = "data"

  def embedchainbot(CHROMA_PATH):
      return App.from_config(
          config={
              "llm": {
                  "provider": "ollama",
                  "config": {
                      "model": "llama3.2:3b",
                      "max_tokens": 250,
                      "temperature": 0.5,
                      "stream": True,
                      "base_url": "http://localhost:11434"  # Ensure this URL is correct and accessible
                  }
              },
              "vectordb": {
                  "provider": "chroma",
                  "config": {"dir": CHROMA_PATH}
              },
              "embedder": {
                  "provider": "ollama",
                  "config": {
                      "model": "nomic-embed-text",
                      "base_url": "http://localhost:11434"  # Ensure this URL is correct and accessible
                  }
              }
          }
      )

  # Add function to display PDF's in the streamlit app
  def display_pdf(file):
      base64_pdf = base64.b64encode(file.read()).decode('utf-8')
      pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="750" type="application/pdf"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)

  def load_documents_with_fallback(DATA_PATH):
      documents = []
      try:
          document_loader = PyPDFDirectoryLoader(DATA_PATH)
          documents = document_loader.load()
      except (PdfReadError, TypeError) as e:
          logger.error(f"Error with PyPDF. Falling back to pdfplumber: {e}")
          with pdfplumber.open(DATA_PATH) as pdf:
              for page_num, page in enumerate(pdf.pages):
                  text = page.extract_text()
                  if text:
                      documents.append(Document(page_content=text, metadata={"source": DATA_PATH, "page": page_num}))
      return documents
                  
  # def load_documents(DATA_PATH):
  #    document_loader = PyPDFDirectoryLoader(DATA_PATH)
  #    try:
  #      documents = []
  #      for doc in document_loader.load():
  #          try:
  #              documents.append(doc)
  #          except PdfReadError as e:
  #              logger.error(f"Error reading PDF: {e}")
  #              continue
  #      logger.info(f"Loaded {len(documents)} documents from {DATA_PATH}")
  #      return documents
  #    except FileNotFoundError as e:
  #        logger.error(f"Error loading documents: {e}")
  #        return []

  def split_documents(documents: list[Document]):
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=800,
          chunk_overlap=80,
          length_function=len,
          is_separator_regex=False,
      )
      chunks = text_splitter.split_documents(documents)
      logger.info(f"Split documents into {len(chunks)} chunks")
      return chunks

  def add_to_chroma(chunks: list[Document], CHROMA_PATH: str):
      try:
          db = Chroma(
              persist_directory=CHROMA_PATH, 
              embedding_function=get_embedding_function()
          )
      except ValueError as e:
          logger.error(f"Error: {e}")
          return

      chunks_with_ids = calculate_chunk_ids(chunks)
      existing_items = db.get(include=[])
      existing_ids = set(existing_items["ids"])
      logger.info(f"Number of existing documents in DB: {len(existing_ids)}")
      print(f"Number of existing documents in DB: {len(existing_ids)}")

      new_chunks = []
      for chunk in chunks_with_ids:
          if chunk.metadata["id"] not in existing_ids:
              new_chunks.append(chunk)

      if new_chunks:
          logger.info(f"Adding new documents: {len(new_chunks)}")
          print(f"👉 Adding new documents: {len(new_chunks)}")
          new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
          db.add_documents(new_chunks, ids=new_chunk_ids)
          db.persist()
          logger.info(f"✅ New documents {new_chunk_ids} added")
          print(f"✅ New documents {new_chunk_ids} added")
      else:
          logger.info("No new documents to add")
          print("✅ No new documents to add")

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
      logger.info(f"Calculated IDs for {len(chunks)} chunks")
      return chunks

  def clear_database(CHROMA_PATH):
      if os.path.exists(CHROMA_PATH):
          try:
              shutil.rmtree(CHROMA_PATH)
          except PermissionError:
              print("File is in use. Retrying in 3 seconds...")
              time.sleep(3)
              shutil.rmtree(CHROMA_PATH)

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
  
      if 'messages' not in st.session_state:
          st.session_state.messages = [] 

      if 'app' not in st.session_state:
          st.session_state.app = embedchainbot(CHROMA_PATH)
      
      parser = argparse.ArgumentParser()
      parser.add_argument("--reset", action="store_true", help="Reset the database.")
      args = parser.parse_args()
      if args.reset:
          print("✨ Clearing Database")
          db = Chroma(persist_directory=CHROMA_PATH)
          db.close()  # Ensure the database connection is closed
          clear_database(CHROMA_PATH)
      
          documents = load_documents_with_fallback(DATA_PATH)
          chunks = split_documents(documents)
          add_to_chroma(chunks, CHROMA_PATH)
          logger.info(f"Added {len(chunks)} documents to the database.")
      
        
      # Load chat history if available
      chat_history_file = 'chat_history.json'
      st.session_state.messages = load_chat_history(log_dir, chat_history_file)
      
      with st.sidebar:
          st.header("Upload PDF")
          pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
          if pdf_file:
              st.subheader("PDF Preview")
              display_pdf(pdf_file)
              
      st.title("Chat with PDF's using LLM's")
      st.caption("This App lets you chat with your model")
      if st.button("Submit PDF"):
          if pdf_file:
              with st.spinner("Adding PDF to the knowledge base..."):
                  if not os.path.exists(DATA_PATH):
                      os.makedirs(DATA_PATH)
                  pdf_path = os.path.join(DATA_PATH, pdf_file.name)
                  with open(pdf_path, 'wb') as f:
                      f.write(pdf_file.getvalue())
                      f.flush()  # Ensure data is written to the file
                      try:
                          st.session_state.app.add(f.name, data_type="pdf_file")
                      except PdfReadError as e:
                          logger.error(f"Error reading PDF: {e}")
                          st.error(f"Error reading PDF: {e}")
                  # os.remove(f.name)
              logger.info(f"Added {pdf_file.name} to the knowledge base!")
              st.success(f"Added {pdf_file.name} to the knowledge base!")
          else:
              logger.error("Please upload a PDF file before submitting.")
              st.error("Please upload a PDF file before submitting.")

      for i, msg in enumerate(st.session_state.messages):
          message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

      if prompt := st.chat_input("Ask a question about the PDF"):
          st.session_state.messages.append({"role": "user", "Content": prompt})
          message(prompt, is_user=True)
          logger.info(f"User: {prompt}")

          save_chat_history(st.session_state.messages, log_dir, chat_history_file)

          with st.spinner("Thinking..."):
              try:
                  response = st.session_state.app.chat(prompt)
                  if isinstance(response, tuple):
                      response = response[0]
                  st.session_state.messages.append({"role": "assistant", "Content": response})
                  message(response)
                  logger.info(f"Chatbot: {response}")
              except ValueError as e:
                  st.error(f"Error: {e}")
                  logger.error(f"Error during chat: {e}")

              save_chat_history(st.session_state.messages, log_dir, chat_history_file)

      if st.button("Clear Chat"):
          st.session_state.messages = []
          st.success("Chat cleared!")
          logger.info("Chat cleared!")

          save_chat_history(st.session_state.messages, log_dir, chat_history_file)

  if __name__ == "__main__":
    main()
    
Chat_With_PDF()