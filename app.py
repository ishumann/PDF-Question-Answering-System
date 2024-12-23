import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# get the text  from each page of pdf/pdfs

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """ Answer the following questions as detailed as possible from the text provided below, make sure to provide all the details, if you don't know the answer, just say "answer is not available in the context", don't provide wrong answers.
    
    
    Context: \n {context}? \n
    Question: \n {question}? \n
    
    
    Answer: 
    
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5,)

    prompt =  PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embedding=embedding)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        { 'input_documents': docs, 'question': user_question}
            , return_only_outputs=True)

    print(response)    

    st.write("Reply: ", response['output_text'])


def main():
    st.set_page_config("Chat With Multiple PDFs")
    st.header("Chat With Multiple PDFs Using Gemini")
    
    user_question = st.text_input("Ask a question from the PDF Files")
    
    
    if user_question:
        user_input(user_question)
        
        
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files, Submit and Process ", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Done")


if __name__ == "__main__":
    main()
