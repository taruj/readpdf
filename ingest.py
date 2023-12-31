## ingest pdf files

import os
from langchain.document_loaders import PDFMinerLoader  # PyPDFLoader, DirectoryLoader,
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS
persist_directory = "db"


def main():

    # Define the directory where the database will be persisted.
    # 'persist_directory' is set to "db", which means the database files will be saved in a directory named 'db'.
    # It's important to ensure this directory exists or is created by the script.
    
    # Iterating over the files in the "docs" directory.
    # os.walk("docs") generates file names in a directory tree by walking the tree either top-down or bottom-up.
    # 'root' is the directory path, 'dirs' is the list of subdirectories, and 'files' is the list of filenames.
    for root, dirs, files in os.walk("docs"):  
        # Processing each file in the directory.
        for file in files:
            # Checking if the file has a .pdf extension.
            if file.endswith(".pdf"):
                print("loading File..."+file)
                # Loading the PDF file using PDFMinerLoader, which extracts text from a PDF file.
                loader = PDFMinerLoader(os.path.join(root, file))

    # Extracting the text from the loaded documents.
    documents = loader.load()

    # Initializing a text splitter.
    # RecursiveCharacterTextSplitter splits text into chunks of a specified size, with an optional overlap.
    # This is useful for processing large texts that need to be broken into smaller parts.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)

    # Splitting the documents into smaller text chunks.
    texts = text_splitter.split_documents(documents)
    
    # Initializing embeddings.
    # SentenceTransformerEmbeddings uses a pre-trained model to generate embeddings from texts.
    # Embeddings are vector representations of text useful for various NLP tasks.
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Creating a Chroma vector store from the documents.
    # Chroma is used for storing and querying vector embeddings.
    # It takes the texts and their embeddings, along with persistence settings.
    db = Chroma.from_documents(texts, embeddings, 
                               persist_directory=persist_directory,  # persist_directory should be defined earlier
                               client_settings=CHROMA_SETTINGS)
 

    # Persisting the database.
    # The 'persist()' method saves the current state of the 'db' object (the Chroma vector store) to disk.
    # This is done in the directory specified by 'persist_directory'.
    # Persisting the database allows for reloading it in future runs, maintaining the state of the embeddings.
    db.persist()


    # Clearing the database from memory.
    db = None

# Ensures that the main function is executed only when the script is run directly.
if __name__ == "__main__":
    main()


