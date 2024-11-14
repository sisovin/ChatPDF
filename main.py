import streamlit as st
from pages.Chat_With_PDF import chat_with_pdf
from pages.Chatbot_RAG_PDFs import chatbot_rag_pdfs

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Chat With PDF", "Chatbot RAG PDFs"])

# Display the selected page
if page == "Chat With PDF":
    st.experimental_set_query_params(page="Chat_With_PDF")
elif page == "Chatbot RAG PDFs":
    st.experimental_set_query_params(page="Chatbot_RAG_PDFs")