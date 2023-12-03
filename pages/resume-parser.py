# Resume-parser contents
import re
import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.prompt import PromptTemplate

from langchain import HuggingFaceHub
from tokenconstant import HUGGINGGACE_API_TOKEN

# for resume-parser


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


def parsing_conv_chat(query, chain, history):
    """
    Funciton to generated answer passing through chain 
    and perform some processing before returning results.
    """
    result = chain.run(query)
    lines = result.strip().split('\n')
    match = re.search(r'Answer:(.*?)Question:', result, re.DOTALL)

    first_answer = ""
    rest_of_questions = []

    i = 0
    while i < len(lines):
        if lines[i].startswith('Answer'):
            if match:
                first_answer = match.group(1).strip()
            else:
                first_answer = lines[i][8:]
            i += 1
            while i < len(lines) and not lines[i].startswith('Question'):
                i += 1
        elif lines[i].startswith('Question'):
            rest_of_questions.append(lines[i][10:])
            i += 1
        else:
            i += 1

    history.append((query, first_answer))

    return first_answer


def display_parsing_history(chain):
    """
    Funciton to display old questions and generated answer.
    """
    reply_container = st.container()
    container = st.container()
    with container:
        user_input = st.chat_input("Ask anything...")
        if user_input:
            with st.spinner('Generating response...'):
                output = parsing_conv_chat(
                    user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:

        with reply_container:
            for i in range(len(st.session_state['generated'])):
                with st.chat_message(name='user', avatar="./aaa.png"):
                    st.write(st.session_state["past"][i])
                with st.chat_message(name="Rp", avatar='./rp-logo.png'):
                    st.write(st.session_state["generated"][i])


def create_retrieval_chain(vector_store):
    '''
    With some standard prompting this funciton returns
    the chain which made with mistrailai model.
    '''
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGGACE_API_TOKEN,
                         repo_id=model_id,

                         model_kwargs={"temperature": 0.1,
                                       "max_new_tokens": 200})
    template = """
    Your task is to help a Hiring manager of the Tech Company.
    The name of the user is given at top part of the document mostly
    so find the name and if the name is asked then return in single word.
    For Education related question get answer from Education portion,
    For projects related question get answer from projects portion,
    For certificates related question get answer from certificates portion,
    For interested areas or hobbies related question get answer
    from Interest or Hobby portion,
    For work experience related question get answer
    from work experience portion,
    For technical Skills related question get answer
    from technical Skills portion,
    The date are present in start to end format for
    Experience and Education portion.
    There may not exist any of above portion then
    return Relevent information is not provided.
    
    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    return qa


def resume_parser():
    '''
    Function that contain mainly ui part and function call.
    '''
    # Initialize session state

    st.markdown(
        """
    <style>
    .title {
        text-align: center;
        margin-bottom: 20px;
        font-size:60px;
    }
    </style>
    """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Resume Parser</h1>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload the Resumes here", accept_multiple_files=True)
    # count = 0

    if uploaded_files:
        initialize_session_state()

        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        # chain = create_conversational_chain(vector_store)
        chain = create_retrieval_chain(vector_store)

        st.markdown(
            """
                <div style='border-top: 2px solid #f2f2f2;
                margin:20px; border-radius: 1px;'>
                """,
            unsafe_allow_html=True
        )

        with st.chat_message(name="Rp", avatar='./rp-logo.png'):
            st.write("Start Parsing The Uploaded Resume.")
        display_parsing_history(chain)

        st.markdown(
            """
                </div>
                """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    resume_parser()
