from langchain_openai import OpenAIEmbeddings
import os
from pinecone import Pinecone
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')

openai.api_key = os.environ["OPENAI_API_KEY"]
model = OpenAIEmbeddings(model="text-embedding-ada-002")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("xppcoder")

from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(
    index, model, "text"
)

def find_match(query):
    result = vectorstore.similarity_search(query, k=10)
    return str(result)

client = openai.OpenAI()

def query_refiner(conversation, query):

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

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string