import os
import base64
import hashlib
import json
import streamlit as st
import httpx
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from streamlit_chat import message
from functions.logger import Logger
from query_data import query_rag, get_book_titles  # Import the query_rag and get_book_titles functions

def Chatbot_RAG_PDFs():
  # Ensure the logs directory exists
  log_dir = 'maniplogs'
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)
      
  # Initialize logger
  logger = Logger(log_file=os.path.join('maniplogs', 'logfile.log'), level=Logger.LEVEL_INFO)

  # Configuration
  CHROMA_PATH = r"D:\Ollama\populatedb"
  model = Ollama(model="llama3.2:3b")  # Replace 'llama3.2:3b' with the actual model name if different
  BASE_URL = "http://localhost:11434"

  DATA_PATH = 'data'

  def test_connection():
      logger.info("Testing connection to the model endpoint...")
      try:
          response = httpx.get(BASE_URL)  # Replace with your model endpoint
          if response.status_code == 200:
              logger.info("Connection successful")
              print("Connection successful")
          else:
              logger.error(f"Connection failed with status code: {response.status_code}")
              print(f"Connection failed with status code: {response.status_code}")
      except Exception as e:
          logger.error(f"Connection failed: {e}")
          print(f"Connection failed: {e}")

  # Ensure the database directory exists
  if not os.path.exists(CHROMA_PATH):
      os.makedirs(CHROMA_PATH)

  # Add function to display PDF's in the streamlit app
  def display_pdf(file):
      logger.info("Displaying PDF...")
      base64_pdf = base64.b64encode(file).decode('utf-8')
      pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="750" type="application/pdf"></iframe>'
      st.markdown(pdf_display, unsafe_allow_html=True)
      logger.info("PDF displayed successfully")

  # Function to save chat history
  def save_chat_history(chat_history, dir, file_name):
      logger.info("Saving chat history...")
      file_path = os.path.join(dir, file_name)
      with open(file_path, 'w') as f:
          json.dump(chat_history, f)
      logger.info("Chat history saved successfully")

  # Function to load chat history
  def load_chat_history(dir, file_name):
      logger.info("Loading chat history...")
      file_path = os.path.join(dir, file_name)
      if os.path.exists(file_path):
          with open(file_path, 'r') as f:
              logger.info("Chat history loaded successfully")
              return json.load(f)
      logger.info("No chat history found")
      return []

  # Save in Hash_file
  def hash_files(files):
      logger.info("Hashing files...")
      hasher = hashlib.md5()
      for file in files:
          file.seek(0)  # Reset file pointer to the beginning
          hasher.update(file.read())
      hash_value = hasher.hexdigest()
      logger.info(f"Files hashed successfully: {hash_value}")
      return hash_value

  # Function to retrieve PDF file by title
  def retrieve_pdf_by_title(title):
      logger.info(f"Retrieving PDF file for title: {title}")
      pdf_path = os.path.join(DATA_PATH, f"{title}.pdf")
      if os.path.exists(pdf_path):
          with open(pdf_path, 'rb') as f:
              logger.info(f"PDF file for title '{title}' retrieved successfully")
              return f.read()
      else:
          logger.error(f"PDF file for title '{title}' not found in {DATA_PATH}")
          st.error(f"PDF file for title '{title}' not found in {DATA_PATH}.")
          return None

  def main():
      logger.info("Starting main function...")
      test_connection()
      # Streamlit title and preparation for the chat
      st.title("Chatbot RAG PDF's using LLM's")
      st.caption("This App lets you chat with your model")

      # Load chat history if available
      chat_history_file = 'chat_history.json'
      st.session_state.messages = load_chat_history(log_dir, chat_history_file)

      # Sidebar for Display the Content 
      with st.sidebar:
          st.header("PDF Title")
          pdf_titles = get_book_titles()
          pdf_titles.insert(0, "Select a book title to discuss")  # Add placeholder option
          pdf_title = st.selectbox("Select a PDF Title", pdf_titles)
          if pdf_title and pdf_title != "Select a book title to discuss":
              st.subheader("PDF Preview")
              pdf_file = retrieve_pdf_by_title(pdf_title)
              if pdf_file:
                  display_pdf(pdf_file)

      # Select Content from knowledge base
      if st.button("Submit PDF"):
          if pdf_title and pdf_title != "Select a book title to discuss":
              with st.spinner("Adding PDF to the knowledge base..."):
                  if not os.path.exists(DATA_PATH):
                      os.makedirs(DATA_PATH)
                  pdf_path = os.path.join(DATA_PATH, f"{pdf_title}.pdf")
                  with open(pdf_path, 'wb') as f:
                      f.write(pdf_file)  # Directly write the bytes object
                      f.flush()  # Ensure data is written to the file
                  logger.info(f"Added {pdf_title} to the knowledge base!")
                  st.success(f"Added {pdf_title} to the knowledge base!")
          else:
              logger.error("Please select a PDF title before submitting.")
              st.error("Please select a PDF title before submitting.")

      # Set Chat Interface
      for i, msg in enumerate(st.session_state.messages):
          message(msg["Content"], is_user=msg["role"] == "user", key=str(i))

      if user_prompt := st.chat_input("Ask a question about the PDF"):
          st.session_state.messages.append({"role": "user", "Content": user_prompt})
          message(user_prompt, is_user=True)
          logger.info(f"User: {user_prompt}")

          # Save chat history
          save_chat_history(st.session_state.messages, log_dir, chat_history_file)

          # Retrieve documents using the query
          response = query_rag(user_prompt)
          if response:
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

      logger.info("Main function completed")

  if __name__ == "__main__":
      main()
      
Chatbot_RAG_PDFs()