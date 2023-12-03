import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
import re
import tempfile
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from tokenconstant import HUGGINGGACE_API_TOKEN
import openai

from langchain.vectorstores import FAISS

# promts
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


def initialize_session_state():
    """
    initialization of session state
    """
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []


def conversation_chat(query, context, chain, history):
    """
    Funciton to generated answer passing through chain. 
    """
    result = chain.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
    # chain({"question": query, "chat_history": history})
    history.append((query, result))
    if result == " ":
        return f"The context doesn't have any information for {query}."
    print("===== type", result)

    answer_part_start = result.find("Answer:")+7
    answer_part_end = result.find("Human:")
    if answer_part_start and answer_part_end:
        return result[answer_part_start:answer_part_end]
    elif answer_part_start:
        return result[answer_part_start:]
    else:
        return result
        # return f"The context doesn't have any information for {query}."
    # return result


def display_chat_history(chain, embeddings, vector_store):
    """
    Funciton to display old questions and generated answer.
    """
    reply_container = st.container()
    container = st.container()
    with container:
        user_input = st.chat_input("Ask anything...")
        if user_input:
            with st.spinner("Generating Response..."):
                context = find_match(user_input, embeddings, vector_store)

                output = conversation_chat(
                    user_input, context, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                with st.chat_message(name='user', avatar="./aaa.png"):
                    st.write(st.session_state["past"][i])
                with st.chat_message(name="Rp", avatar='./rp-logo.png'):
                    st.write(st.session_state["generated"][i])


def find_match(input_text, embeddings, vector_store):
    '''
    Function in order to find the match for given input in vector store.
    '''
    input_embedding = embeddings.embed_query(input_text)
    docs = vector_store.similarity_search_by_vector(input_embedding, k=3)
    result = ""
    for doc in docs:
        result += doc.page_content + "\n"
    return result


def create_conversational_chain():
    '''
    Initializing LLM, Defining some prompt templates
    and also creating a ConversationChain
    the chain which made with mistrailai model.
    '''
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGGACE_API_TOKEN,
                         repo_id=model_id,
                         model_kwargs={"temperature": 0.1,
                                       "max_new_tokens": 200})

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k=3, return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""
    [INST]
        You are a helpful trekking information assistant based
        on given context.
        Your task is to generate a single answer without providing
        explanations.
    [/INST]
    '""")

    human_msg_template = HumanMessagePromptTemplate.from_template(
        template="{input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [system_msg_template,
         MessagesPlaceholder(variable_name="history"),
         human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory,
                                     prompt=prompt_template, llm=llm,
                                     verbose=True)

    return conversation


def home():
    st.title("Trekking Info Nepal")
    initialize_session_state()

    loader = TextLoader("documents/trekkingdata.txt")
    txtdata = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20)

    text_chunks = text_splitter.split_documents(txtdata)
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain()

    st.markdown(
        """
            <div style='border-top: 2px solid #f2f2f2; margin:20px; border-radius: 1px;'>
            """,
        unsafe_allow_html=True
    )

    with st.chat_message(name="Rp", avatar='./rp-logo.png'):
        st.write("Start Talking Chatbot that use Mistral-7b-instruct Model ")
    display_chat_history(chain, embeddings, vector_store)

    st.markdown(
        """
            </div>
            """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    home()
