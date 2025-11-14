import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser 


DATA_PATH = "documents"
CHROMA_PATH = "chroma"


st.set_page_config(page_title="AI Personal Knowledge Base", layout="wide")
st.title(" AI-Powered Personal Knowledge Base")
st.markdown("Ask questions about the content in your 'documents' folder.")

if not os.path.exists(CHROMA_PATH):
    st.info("Chroma database not found. Starting ingestion process...")
    
    with st.spinner("Loading and splitting documents..."):
       
        loader = DirectoryLoader(DATA_PATH, glob="**/*[.pdf,.txt]")
        documents = loader.load()
        
     
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("No text could be extracted from the documents. Please ensure they are text-based.")
        else:
            st.success(f"Split documents into {len(chunks)} chunks.")
            
           
            with st.spinner("Creating embeddings and storing in Chroma DB... This may take a moment."):
                embeddings = OllamaEmbeddings(model="llama3")
                db = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embeddings, 
                    persist_directory=CHROMA_PATH
                )
                st.success("✅ Database created successfully!")


query_text = st.text_input("Ask a question:", placeholder="What did I write about...")

if query_text:
    with st.spinner("Searching the knowledge base and thinking..."):
        try:
           
            embeddings = OllamaEmbeddings(model='llama3')
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            retriever = db.as_retriever()
            model = Ollama(model="llama3")

       
            prompt_template = """
            Answer the question based only on the following context:
            {context}
            ---
            Answer the question based on the above context: {question}
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
        
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

        
            response = chain.invoke(query_text)
            st.success("Here's the answer:")
            st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {e}")