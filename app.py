# pip install InstructorEmbedding

# pip install sentence_transformers
import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        temperature=0,
        model="microsoft/Phi-3-mini-4k-instruct",
        api_key="")

    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),
                     unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),
                     unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat With Paper", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Enter Your Questions")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Research Paper")
        pdf_docs = st.file_uploader("Upload The Research Paper Here",
                                    accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_splitter = CharacterTextSplitter(separator="\n",
                                                      chunk_size=1000,
                                                      chunk_overlap=200,
                                                      length_function=len)
                text_chunks = text_splitter.split_text(text)

                # create vector store

                instructor_embeddings = HuggingFaceInstructEmbeddings(
                    model_name="hkunlp/instructor-xl",
                    model_kwargs={"device": "cuda"})

                vectorstore = FAISS.from_texts(texts=text_chunks,
                                               embedding=instructor_embeddings)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
