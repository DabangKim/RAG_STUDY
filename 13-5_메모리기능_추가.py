# 0. 필수 라이브러리 호출
import os
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory



# 1. PDF 문서 로드 및 벡터화 함수

# 1-1 PDF 문서 로드 함수
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# 1-2 Document 객체를 ChromaDB에 저장하는 함수
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

# 1-3 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
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



# 2. RAG 체인 구성
@st.cache_resource
def initialize_components():
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

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """
    대화 기록과 최신 사용자 질문이 주어졌을 때, 해당 질문이 대화 기록의 맥락을 참조하고 있을 수 있습니다. \
    대화 기록 없이도 이해할 수 있도록 질문을 독립적인 형태로 재구성하세요. \
    필요하지 않다면 원래 질문을 그대로 반환하세요. \
    절대로 질문에 답하지 마세요.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문-답변 시스템 프롬프트
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
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever_from_llm, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain



# 3. Streamlit UI 구성
st.title("헌법 Q&A 챗봇 💬 📚")
rag_chain = initialize_components()

# 3-1 세션 ID 생성 및 보관
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# 3-2 세션별로 히스토리 저장 딕셔너리 선언 (Streamlit 세션 상태 내에 저장)
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

# 3-3 session_id -> StreamlitChatMessageHistory 연결 함수
def get_session_history(session_id: str):
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = StreamlitChatMessageHistory(
            key=f"chat_messages_{session_id}"
        )
    return st.session_state.history_store[session_id]
chat_history = get_session_history(session_id)

# 3-4 RunnableWithMessageHistory 구성
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 3-5 최초 시스템 메시지 출력
if not chat_history.messages:
    chat_history.add_ai_message("헌법에 대해 무엇이든 물어보세요!")

# 3-6 이전 채팅 메시지 불러오기
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 3-7 사용자 입력 받기
if prompt_message := st.chat_input("질문을 입력해주세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("생각 중입니다..."):
            config = {"configurable": {"session_id": session_id}}
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )
            answer = response['answer']
            st.write(answer)