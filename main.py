import streamlit as st
from streamlit_chat import message
from utils import initialize_services, find_match, query_refiner, get_conversation_string
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

st.subheader("X++ Coding Assistant")

# Load API keys from the URL query parameters (i.e., cookies)
if "openai_api_key" in st.query_params:
    st.session_state["openai_api_key"] = st.query_params["openai_api_key"]
if "pinecone_api_key" in st.query_params:
    st.session_state["pinecone_api_key"] = st.query_params["pinecone_api_key"]

# Check if API keys are already saved in session state
if "openai_api_key" not in st.session_state or "pinecone_api_key" not in st.session_state:
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    pinecone_api_key = st.text_input("Enter your Pinecone API key:", type="password")

    if st.button("Save API Keys"):
        st.session_state["openai_api_key"] = openai_api_key
        st.session_state["pinecone_api_key"] = pinecone_api_key

        # Save the API keys in query parameters using a dictionary
        st.query_params["openai_api_key"] = openai_api_key
        st.query_params["pinecone_api_key"] = pinecone_api_key
        st.success("API keys saved!")

else:
    st.write("API keys already saved.")

# Use the saved API keys
if "openai_api_key" in st.session_state and "pinecone_api_key" in st.session_state:
    vectorstore, client = initialize_services(
        st.session_state["openai_api_key"],
        st.session_state["pinecone_api_key"]
    )

    # Dropdown for selecting the chat LLM model with a default value of "gpt-3.5-turbo"
    model_name = st.selectbox(
        "Choose the chat model for the LLM:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"],
        index=0  # Setting "gpt-3.5-turbo" as the default option
    )

    # Initialize services with the provided OpenAI and Pinecone API keys
    vectorstore, client = initialize_services(st.session_state["openai_api_key"], st.session_state["pinecone_api_key"])

    # Use the selected model from the dropdown
    llm = ChatOpenAI(model_name=model_name, openai_api_key=st.session_state["openai_api_key"])

    if 'responses' not in st.session_state:
         st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an assistant for Dynamics 365 Finance and Operations. Answer the question as truthfully as possible using the provided context,
    and if the answer is not contained within the text below, say 'I don't have information regarding this'. If the user asks to do something with code consult the text below and write a code snippet that would solve the user's problem in X++. Make sure the code is written in the new Dynamics 365 Finance and Operations standard and DO NOT USE code from Dynamics AX2012.""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(client, conversation_string, query)
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = find_match(vectorstore, refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                st.write(st.session_state['responses'][i])  # Replace message with st.write
                if i < len(st.session_state['requests']):
                    st.write(st.session_state["requests"][i])
else:
    st.write("Please enter your API keys to continue.")