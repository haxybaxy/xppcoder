from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import streamlit as st

# Load the environment variables from the .env file
load_dotenv()
def initialize_services(openai_api_key, pinecone_api_key):
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("xppcoder")

    # Set up Pinecone VectorStore
    vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    client = openai.OpenAI()

    return vectorstore, client

def find_match(vectorstore, query):
    result = vectorstore.similarity_search(query, k=10)
    return str(result)

def query_refiner(client, conversation, query):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string