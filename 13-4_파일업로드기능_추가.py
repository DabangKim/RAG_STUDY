# 0. 필수 라이브러리 호출

import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 1. 업로드된 PDF 문서 처리
@st.cache_resource
def load_pdf(_file):
    # 임시 파일을 생성하여 업로드된 PDF 파일의 데이터를 저장
    # tempfile : 임시 파일을 만들 때 사용하는 python 표준 라이브러리
    # NamedTemporaryFile : 이름이 있는 임시 파일을 생성
    # mode="wb" : 파일을 쓰기 모드(w)로 열되, 바이너리 모드(b)로 열기 -> 이미지, 오디오, PDF 등과 같은 바이너리 파일을 다룰 때 사용
    # delete=False : 임시 파일을 닫아도 삭제하지 않도록 설정
    # delete=True로 설정하면, 파일을 닫을 때 자동으로 삭제됨
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        # 업로드된 파일의 내용을 임시 파일에 기록
        tmp_file.write(_file.getvalue())
        # 임시 파일의 경로를 변수에 저장
        tmp_file_path = tmp_file.name
        # 임시 파일의 데이터를 로드
        loader = PyPDFLoader(file_path=tmp_file_path)
    return loader.load()

@st.cache_resource
def create_vector_store(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    text_splitter = SemanticChunker(embeddings)
    split_docs = text_splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    return vectorstore

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


# 2. RAG 체인 구성
@st.cache_resource
def chaining(_pages):
    vectorstore = create_vector_store(_pages)

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
    당신은 질문 응답 작업을 위한 어시스턴트입니다. \
    아래에 제공된 검색된 문맥 정보를 활용하여 질문에 답해주세요. \
    답을 모를 경우, 모른다고 솔직하게 말해주세요. \
    답변은 한국어로 정중하게 작성해주세요.\
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
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
st.header("ChatPDF 💬 📚")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])  # st.file_uploader("파일 업로더의 제목", type=["원하는 타입"])
if uploaded_file is not None:
    pages = load_pdf(uploaded_file)

    rag_chain = chaining(pages)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

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