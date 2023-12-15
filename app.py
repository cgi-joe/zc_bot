import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
import transformers
from langchain.prompts import PromptTemplate



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def setup_chain(vector_store, api_key):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    
    template = """You are a representative customer service specialist at Zero Commission Real Estate.
    Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you know, don't try to make up an answer.
    If question is relevant to Real Estate, but you do not have answer, say that you will ask a real estate specialist at Zero Commission.

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else. If the question is not relevant to the context, say politely you cannot help.
    If you know the exact response to the question, reply with the full response. Add any other helpful and relevant info that may be helpful if you deem it appropriate to the question.
    Helpful answer:
    """
    
    qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': qa_prompt})


def get_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    start_time = time.time()
    print("Starting query...")  
    response = st.session_state.conversation_chain({'question': user_question})
    query_time = time.time() - start_time 
    st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)
    st.write(f"Query Time: {query_time:.2f} seconds") 
    print(f"Query Time: {query_time:.2f} seconds") 
    
def handle_user_input2(user_question):
    start_time = time.time()
    print("Starting query...")  
    response = st.session_state.conversation_chain({'query': user_question})
    query_time = time.time() - start_time 
    st.write(bot_template.replace("{{MSG}}", response['result']), unsafe_allow_html=True)
    st.write(f"Query Time: {query_time:.2f} seconds") 
    print(f"Query Time: {query_time:.2f} seconds") 

def main():
    st.set_page_config(page_title="Zero Commission Assistant", page_icon=":chat:")
    st.write(css, unsafe_allow_html=True)

    st.header("Zero Commission Assistant")

    # Input for OpenAI API key
    with st.sidebar:
        st.subheader("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key")
        if st.button("Add Key"):
            st.session_state['openai_api_key'] = api_key
            st.success("API Key added successfully!")

    user_question = st.text_input("Ask a question about Zero Commission Real Estate!")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if 'openai_api_key' not in st.session_state or not st.session_state['openai_api_key']:
                st.error("Please add your OpenAI API Key first.")
            else:
                with st.spinner("Processing"):
                    start_time = time.time()
                    print("Starting processing...")  
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks, st.session_state['openai_api_key'])
                    st.session_state.conversation_chain = setup_chain(vectorstore, st.session_state['openai_api_key'])
                processing_time = time.time() - start_time  
                st.sidebar.write(f"Processing Time: {processing_time:.2f} seconds") 
                print(f"Processing Time: {processing_time:.2f} seconds") 


    if user_question and "conversation_chain" in st.session_state:
        handle_user_input2(user_question)

if __name__ == '__main__':
    main()