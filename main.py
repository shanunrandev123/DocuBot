import streamlit as st
import pandas as pd

from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import CharacterTextSplitter


from dotenv import load_dotenv
import io



load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text


# parameter chunk overlap helps in retaining the semantic context between the chunks 

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split(text)
#     return chunks


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunks = text_splitter.split_text(text)

    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    
    vector_store.save_local('faiss_index')
    
def get_conversational_chain():
    
    prompt_template = """
    
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not there in the provided context just say "answer is not available in the provided context", dont give the wrong answer
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    
    
    """

    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.3)
    
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context', 'question'])
    
    chain = load_qa_chain(llm = model, chain_type = 'stuff', prompt = prompt)
    
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    response = chain({'input_documents': docs, 'question': user_question}, return_only_outputs = True)
    
    print(response)
    
    st.write('Reply: ', response['output_text'])
    
    
    
def run():
    st.set_page_config(page_title = 'Chat with multiple PDFs')
    st.header('Chat with multiple PDFs using Gemini')
    
    user_question = st.text_input('Ask your question here')
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title('Menu:')
        
        pdf_docs = st.file_uploader('Upload PDFs and click on Submit & process', type = 'pdf', accept_multiple_files = True)
        if st.button('Submit & process'):
            with st.spinner('Processing PDFs...'):
                
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success('PDFs processed successfully')
                
                
                
if __name__ == '__main__':
    run()
    



