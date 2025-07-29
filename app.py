#General Basic AI LLM Wrapper

import os
from langchain_community.llms import Ollama
import streamlit as st 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Loading Env

load_dotenv()

#Langchain Tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

#Prompt Template

prompt = ChatPromptTemplate(
    [
        ("system", "You're an AI & Politics Expert. Please help me breakdown my questions and provide proper answers."),
        ("user","Question:{question}")
    ]
)

#Streamlit Framework

st.title("Langchain Project RAG with Gemma")
input_text = st.text_input("What's up?")

#Using Ollama

llm = Ollama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))