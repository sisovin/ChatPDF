import os
import httpx
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from functions.logger import Logger

# Ensure the logs directory exists
log_dir = 'maniplogs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

CHROMA_PATH = r"D:\Ollama\populatedb"
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    logger.info(f"Querying RAG with text: {query_text}")
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=300)

    if not results:
        logger.info("No results found.")
        return "No results found."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # logger.info(f"Context text: {context_text}")  # Commented out to avoid logging extensive context text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # logger.info(f"Prompt: {prompt}")  # Commented out to avoid logging the prompt

    try:
        # Use the actual model for querying with additional parameters
        model = Ollama(model="llama3.2:3b")  # Replace 'llama3.2:3b' with the actual model name if different
        response_text = model.invoke(
            prompt,
            max_tokens=250,
            temperature=0.8,
            stream=True
        )
        logger.info(f"Response text: {response_text}")

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        # formatted_response = f"Response: {response_text}\nSources: {sources}"
        # logger.info(f"Formatted response: {formatted_response}")  # Commented out to avoid logging the formatted response
        
        # Extract and clean book titles from sources
        book_titles = set()
        for source in sources:
            if source:
                book_title = os.path.basename(source).split(':')[0]
                book_titles.add(book_title.replace('_', ' ').replace('.pdf', ''))

        # Format book titles as a numbered list
        book_titles_list = list(book_titles)
        formatted_book_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(book_titles_list)])

        print(f"Book Titles:\n{formatted_book_titles}")
        logger.info(f"Book Titles:\n{formatted_book_titles}")

        return response_text
      
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return f"Error invoking model: {e}"
      
def test_connection():
    try:
        response = httpx.get("http://localhost:11434")  # Replace with your model endpoint
        if response.status_code == 200:
            print("Connection successful")
        else:
            print(f"Connection failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
        
def get_book_titles():
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Retrieve all unique book titles from the database
    results = db.similarity_search_with_score("", k=300)
    book_titles = set()
    for doc, _score in results:
        source = doc.metadata.get("id", None)
        if source:
            book_title = os.path.basename(source).split(':')[0]
            book_titles.add(book_title.replace('_', ' ').replace('.pdf', ''))
    
    return list(book_titles)

if __name__ == "__main__":
    test_connection()