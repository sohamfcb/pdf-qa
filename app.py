import google.generativeai as genai
import streamlit as st
import os
from PyPDF2 import PdfReader
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI API with the provided API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def getPDFText(pdf_docs):
    """
    Extract text from the uploaded PDF documents.

    Args:
        pdf_docs (list): List of uploaded PDF files.

    Returns:
        str: Combined text extracted from all PDF documents.
    """
    text = ''
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ''  # Handle None if text extraction fails
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")

    return text

def getTextChunks(text):
    """
    Split the provided text into manageable chunks for processing.

    Args:
        text (str): Input text to be split.

    Returns:
        list: List of text chunks.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return []

def getVectorStore(text_chunks):
    """
    Create and save a vector store from text chunks using embeddings.

    Args:
        text_chunks (list): List of text chunks to be converted into embeddings.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local('faiss_index')
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def getConversationalChain():
    """
    Create a conversational chain for question answering.

    Returns:
        Chain: A chain object for processing questions.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    try:
        model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-exp-0801', temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")

def user_input(user_question):
    """
    Process the user question and return a response based on the PDF context.

    Args:
        user_question (str): The question input by the user.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = getConversationalChain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        output = response['output_text']
        st.markdown(output)
    except Exception as e:
        st.error(f"Error processing user input: {e}")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Ask PDF")
    st.header("Chat with PDFs")  

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the \"Submit & Process\" Button", accept_multiple_files=True)

        btn1 = st.button("Submit & Process")

        if pdf_docs:
            if btn1:
                with st.spinner("Processing..."):
                    raw_text = getPDFText(pdf_docs)
                    text_chunks = getTextChunks(raw_text)
                    getVectorStore(text_chunks)
                    st.success("Done")
        
        elif btn1 and pdf_docs is None:
            st.error("Please enter at least one PDF file!")

    user_question = st.text_area("Ask your Question")

    btn = st.button('Ask')
    if btn:
        if user_question:
            user_input(user_question)

if __name__ == '__main__':
    main()
