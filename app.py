import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from docx import Document
import io


email_link = '<a href="mailto:ravinell@zerocommission.com">ravinell@zerocommission.com</a>'

default_template = """You are a representative customer service specialist at Zero Commission Real Estate.
Use the following pieces of information to answer the user's question.

Context:{context}
Question:{question}

Do not repeat the question in your response. If you have the exact response, repeat it exactly.
If you do not know the answer or the question is irrelevant, politely say you cannot help. 
If you know the exact response to the question, reply with the full response. If there is additional information helpful in the response, include it.
If you do not know the answer, but the question is relevant, you can ask them to contact ravinell@zerocommission.com.
For further answers, always tell them to contact ravinell@zerocommission.com.
Helpful answer:
"""



def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.type == "application/pdf":
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
        elif doc.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            docx = Document(doc)
            for para in docx.paragraphs:
                text += para.text + '\n'
        elif doc.type == "text/plain":
            doc.seek(0)
            text += doc.read().decode('utf-8') + '\n'
        else:
            raise ValueError(f"Unsupported file type: {doc.type}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=7000,
        chunk_overlap=1000,
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
    
    template = st.session_state['template']
    
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
    response = st.session_state.conversation_chain({'query': user_question})
    query_time = time.time() - start_time
    
    # Ensure HTML in the response is rendered correctly
    formatted_response = bot_template.replace("{{MSG}}", response['result']).replace("ravinell@zerocommission.com", email_link)
    st.markdown(formatted_response, unsafe_allow_html=True)
    
    st.write(f"Query Time: {query_time:.2f} seconds") 
    print(f"Query Time: {query_time:.2f} seconds")

def main():
    st.set_page_config(page_title="Zero Commission Assistant", page_icon=":chat:")
    st.write(css, unsafe_allow_html=True)

    st.header("Zero Commission Assistant")

    # Initialize the template in session state if not already present
    if 'template' not in st.session_state:
        st.session_state['template'] = default_template

    # Input for OpenAI API key
    with st.sidebar:
        st.subheader("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key")
        if st.button("Add Key"):
            st.session_state['openai_api_key'] = api_key
            st.success("API Key added successfully!")


    
    with st.form(key='user_query_form'):
        user_question = st.text_input("Ask a question about Zero Commission Real Estate!")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and "conversation_chain" in st.session_state:
        handle_user_input(user_question)
    
    st.subheader("Template Editing")
    if st.button("Edit Template"):
        st.session_state['edit_mode'] = True

    if st.session_state.get('edit_mode', False):
        new_template = st.text_area("Edit the Template:", st.session_state['template'], height=300)
        if st.button("Save Template"):
            st.session_state['template'] = new_template
            st.session_state['edit_mode'] = False
            st.success("Template updated!")
            st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore, st.session_state['openai_api_key'])
            st.experimental_rerun()
        
        if st.button("Reset to Default Template"):
            st.session_state['template'] = default_template
            st.success("Template reset to default!")
            st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore, st.session_state['openai_api_key'])
            st.experimental_rerun()


    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your documents here and click on 'Process'", 
                        accept_multiple_files=True, 
                        type=['pdf', 'docx', 'doc', 'txt'])
        if st.button("Process"):
            if 'openai_api_key' not in st.session_state or not st.session_state['openai_api_key']:
                st.error("Please add your OpenAI API Key first.")
            else:
                try:
                    with st.spinner("Processing"):
                        start_time = time.time()
                        print("Starting processing...")
                        if 'raw_text' not in st.session_state:
                            st.session_state.raw_text = get_document_text(docs)
                        st.session_state.raw_text = get_document_text(docs)
                        if 'text_chunks' not in st.session_state:
                            st.session_state.text_chunks = get_text_chunks(st.session_state.raw_text)
                        st.session_state.text_chunks = get_text_chunks(st.session_state.raw_text)
                        if 'vectorstore' not in st.session_state:
                            st.session_state.vectorstore = get_vectorstore(st.session_state.text_chunks, st.session_state['openai_api_key'])
                        st.session_state.vectorstore = get_vectorstore(st.session_state.text_chunks, st.session_state['openai_api_key'])
                        if 'conversation_chain' not in st.session_state:
                            st.session_state.conversation_chain = setup_chain(st.session_state.vectorstore, st.session_state['openai_api_key'])
                    processing_time = time.time() - start_time  
                    st.sidebar.write(f"Processing Time: {processing_time:.2f} seconds") 
                    print(f"Processing Time: {processing_time:.2f} seconds")
                except ValueError as e:
                    st.error(str(e))
                    
if __name__ == '__main__':
    main()