import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
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
DATA_PATH = "data"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def main():
    logger.info("Starting the database population process.")

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logger.info("Reset flag detected. Clearing the database.")
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    logger.info("Loading documents.")
    documents = load_documents()
    logger.info(f"Loaded {len(documents)} documents.")

    logger.info("Splitting documents into chunks.")
    chunks = split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    logger.info("Adding chunks to Chroma.")
    add_to_chroma(chunks)
    logger.info("Finished adding chunks to Chroma.")

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        logger.info(f"👉 Adding new documents: {len(new_chunks)}") 
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        # Split new_chunks into smaller batches
        batch_size = 5000  # Adjust batch size as needed
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i + batch_size]
            batch_ids = new_chunk_ids[i:i + batch_size]
            try:
                db.add_documents(batch_chunks, ids=batch_ids)
                logger.info(f"Added batch {i // batch_size + 1} with {len(batch_chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error adding documents to Chroma: {e}")
                print(f"Error adding documents to Chroma: {e}")
    else:
        print("✅ No new documents to add")
        logger.info("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/{documentName}.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
        
    logger.info(f"Number of chunks: {len(chunks)}")
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("Database cleared successfully.")

if __name__ == "__main__":
    main()