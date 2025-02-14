# *************************************************************************
#*  ChatGPT thread which helped me write code : (Ollama LLM vs ChatOllama)
#*  https://chatgpt.com/share/67ab2f5d-4af0-8005-8439-c8a1c7df993a   
# *************************************************************************


import streamlit as st
import random
import time
import os
import uuid

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Chroma vector store wrapper

# For conversation memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

#1. chat interface
#2. llm integration q&a
#3. rag implementation

# =============================================================================
# Setup ChromaDB (in-memory, no persistence)
# =============================================================================

# Here, we simply don't provide a persist_directory so that Chroma runs in-memory.
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
# from chromadb.config import Settings

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# Initialize the vector store with your ephemeral client.
client = chromadb.EphemeralClient(settings=Settings(), tenant=DEFAULT_TENANT, database=DEFAULT_DATABASE, )
vector_store = Chroma(
    client=client,
    collection_name="example_collection",
    embedding_function=embedding_model,
)

# Define a simple parser that extracts the text from an AIMessage
class SimpleParser:
    def __call__(self, response):
        # If the response is an AIMessage, return its content (strip any extra whitespace)
        if isinstance(response, AIMessage):
            print('hahaha')
            return response.content.strip()
        # Otherwise, assume it's a string
        return str(response).strip()


#######################################
# Functions for File Processing and RAG
#######################################


def extract_text(file):
    """
    Replace this with your actual text extraction logic.
    For example, for PDFs you might use PyMuPDF or pdfminer.
    """

    # In this placeholder, we simply decode the file contents.
    return file.read().decode("utf-8", errors="ignore")

def chunk_text(text):
    """

        """
    my_custom_separators = [
        r'\d+\.\s',              # Matches numbered rules like "1. " or "10. "
        r'\n\n',                 # Double newline for paragraphs
        r'\n',                   # Single newline
        r'(?<=\.)\s+(?=[A-Z])',   # Matches a sentence-ending period (followed by space and an uppercase letter)
        r' ',                    # Space
        ""                      # Fallback: empty string (matches every character)
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500, 
        chunk_overlap=20,
        separators=my_custom_separators,
        is_separator_regex=True
    )


    chunks = text_splitter.create_documents([text])
    print("No of chunks after splitting: ", len(chunks), '\n\n')
    print("chunk sample:", chunks[0].page_content)
    return chunks # chunks is list of langchain documents . to get text use chunk.page_content method



def embed_store_embeddings(chunks, source_filename):
     
    """
    Store the document chunks and their embeddings in Chroma.
    Adds metadata "source": filename to each document.
    """
    
    # Add metadata to each document.
    for doc in chunks:
        if doc.metadata:
            doc.metadata["source"] = source_filename
        else:
            doc.metadata = {"source": source_filename}
    uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)
    

def delete_embeddings_by_source(filename):
    """
    Delete all documents from the vector store with metadata matching {"source": filename}.
    """
    # Chroma's delete method supports filters.
    vector_store.delete_documents(filter={"source": filename})
    st.sidebar.write(f"Embeddings for {filename} deleted.")


def process_file(uploaded_file, progress_callback):
    
    """
    Process the uploaded file with real steps, updating the UI via the callback.
    steps:
      1. Extract text,
      2. Chunk the text,
      3. Compute embeddings,
      4. Store embeddings in Chroma.
    Args:
        uploaded_file: The file uploaded by the user.
        progress_callback: A function that takes (percentage, message)
                           to update the UI in real time.
    Returns:
        A success message upon completion.
    """
    
    print('file name : ', uploaded_file.name)
    # Step 1: Extract text
    progress_callback(0, "Starting text extraction...")
    text = extract_text(uploaded_file)
    
    progress_callback(25, "Text extraction complete.")

    # Step 2: Chunk the text
    chunks = chunk_text(text)
    progress_callback(50, f"Chunking complete. {len(chunks)} chunks created.")

    # Step 3: Create embeddings
    embed_store_embeddings(chunks, uploaded_file.name)


    progress_callback(100, "Embeddings created and stored successfully.")

    return "File processed and embeddings created for RAG!"


def ask_ai(user_input):
    
    results = vector_store.similarity_search(
        user_input,
        k=5,
    )
    
    context = '\n\n'.join([doc.page_content for doc in results])
    system_message = f"You are a helpful assistant that recommends code snippets based on given context:\n\n{context}\n\nand human input description of code snippet."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", user_input)
        ]
    )


    llm = ChatOllama(
        model="qwen2.5-coder:0.5b",
        temperature=0,
        # other params...
    )
    
    # Instantiate the parser.
    parser = SimpleParser()
    
    chain = prompt | llm | parser
    
    ai_msg = chain.stream({
        'human_input':{user_input}
    })
    
    return ai_msg



if True:
    st.header("{ } search code snippet 💡")


    # st.sidebar.header('Add your pdf, txt, md files')
    uploaded_file = st.sidebar.file_uploader('Upload your  file', type=['pdf', 'txt', 'md'], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")
    if upload  := st.sidebar.button('Process file'):
        
        if uploaded_file is not None:
            # Create Progressbar element
            progress_bar = st.sidebar.progress(0)
            progress_text = st.sidebar.empty()
            
            # define a callback function to update the progress
            def progress_callback(percentage, message):
                progress_bar.progress(percentage)
                progress_text.text(message)
            
            result_message = process_file(uploaded_file, progress_callback)
            st.sidebar.success(result_message) 
        else:
            st.sidebar.error("Please upload a file before clicking submit.")



##################################################################################
# code which loads documents folder into vectorstore when file checbox is checked
##################################################################################


import os

# Callback function for checkbox changes.
def checkbox_onchange():
    # Loop through all documents to check their state
    for filename in os.listdir('./documents/'):
        if filename.endswith((".txt", ".md")):
            # Each checkbox's value is stored in st.session_state using the filename as key.
            value = st.session_state.get(filename, False)
            # st.write(f"{filename}: {value}")
            
            # if value == False:
                # code to remove  embeddings from vectorstore using metadata . source : filename
                

## the below piece code will display files from documents folder in sidebar.
for filename in sorted(filter(lambda filename: filename.endswith((".txt", ".md")), os.listdir('./documents/') )):
    
    if checked := st.sidebar.checkbox(filename, key=filename, value=False, on_change=checkbox_onchange):
        
        # save the file in vector store
        ## 1. read file
        ## 2. splitting
        ## 3. embedding 
        ## 4. saving
        
        # 1.
        
        with open(os.path.join('documents', filename), 'rb') as file:
            text = extract_text(file)
        
            # progress_callback(25, "Text extraction complete.")

            # Step 2: Chunk the text
            chunks = chunk_text(text)
            # progress_callback(50, f"Chunking complete. {len(chunks)} chunks created.")

            # Step 3: Create embeddings
            embed_store_embeddings(chunks, filename)
        
            st.sidebar.write(f'{filename} is ready to use ')


# st.write(st.session_state)

# Initialize chat history  
# create a empyt list  of messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
         st.markdown(message["content"])



        
        
# Take user input
user_input = st.chat_input("What is up?")

if user_input:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    
    # Display response from ai assistant
    with st.chat_message('assistant'):
        response = st.write_stream(ask_ai(user_input))
        
    # save the response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    



