# 0. 필수 라이브러리 호출
import os
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


# 1. PDF 문서 로드 및 벡터화 함수
## PDF 문서 로드 함수
## @st.cache_resource : Streamlit에서 사용하는 데코레이터. 리소스를 한 번만 생성해서 캐시해두고 다시 실행할 때 재사용할 수 있도록 도와주는 기능
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

## Document 객체를 ChromaDB에 저장하는 함수
@st.cache_resource
def create_vector_store(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    text_splitter = SemanticChunker(embeddings)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs, 
        embeddings,
        persist_directory = persist_directory
    )
    return vectorstore

## 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            embedding_function = embeddings,
            persist_directory = persist_directory
        )
    else:
        return create_vector_store(_docs)

## Document 객체의 page_content를 Join하여 하나의 문자열로 변환하는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 2. RAG 체인 구성
## RAG 체인 생성 함수
@st.cache_resource
def chaining():
    file_path = "data/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    llm = ChatOllama(
        model = "basic_kanana",
        temperature = 0.7
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs = {"lambda_mult": 0.7, "fetch_k": 10, "k": 3}
        ),
        llm = llm
    )

    qa_system_prompt = """
    당신은 질의응답 작업을 위한 어시스턴트입니다. \
    다음에 제공된 문맥 정보를 활용하여 질문에 답하십시오. \
    답을 모를 경우, 모른다고만 답하십시오. \
    최대 세 문장으로 간결하게 답변하십시오. \
    존댓말로 대답하십시오. \
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}")
        ]
    )

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# 3. Streamlit UI 구성
st.header("헌법 Q&A 챗봇 💬 📚")
rag_chain = chaining()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)