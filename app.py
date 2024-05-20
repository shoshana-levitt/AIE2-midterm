# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# Import dependencies
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import chainlit as cl
from langchain.retrievers import MultiQueryRetriever

# Get API key
load_dotenv()

# Load Data
loader = PyMuPDFLoader(
    "/Users/SKTL/Desktop/Coding/VS-Code/AIE2-midterm/meta_pdf.pdf",
)
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 100
)
documents = text_splitter.split_documents(documents)

# Load OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create QDrant VectorStore
qdrant_vector_store = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="meta-10-k-filings",
)

# Create Retriever
retriever = qdrant_vector_store.as_retriever()

# Template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

# Chainlit App
@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

@cl.on_message
async def main(message: cl.Message):
    chainlit_question = message.content
    response = retrieval_augmented_qa_chain.invoke({"question": chainlit_question})
    chainlit_answer = response["response"].content
    msg = cl.Message(content=chainlit_answer)
    await msg.send()
