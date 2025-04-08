import streamlit as st
from langchain_ollama import ChatOllama

# 웹 페이지 제목 설정
st.title("💬 Chatbot")

# 세션 상태(session_state)에 'messages' 키가 없으면 초기화
# session_state는 streamlit의 상태 관리 기능으로, 사용자와의 상호작용 간에 데이터를 유지하기 위해 사용됨
# 최초 실행 시, 챗봇이 먼저 인사말을 하도록 기본 메시지를 세팅함
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "도움이 필요하신가요?"}]

# 지금까지 대화 내용을 화면에 출력
# 사용자와 챗봇이 주고받은 메시지를 순서대로 화면에 렌더링
for msg in st.session_state.messages:
    # msg["role"]이 'user' 또는 'assistant'일 경우 각각 사용자/AI 메시지로 렌더링됨
    # chat_message() 함수는 사용자 또는 AI의 입력값을 채팅 컨테이너에 표시하는 역할을 함
    st.chat_message(msg["role"]).write(msg["content"])
    # # 다른 방식
    # with st.chat_message(msg["role"]):
    #     st.write(msg["content"]

# 사용할 AI 모델을 정의
# model은 카카오의 'kanana' 모델을 사용하고, temperature는 창의성(랜덤성)을 0.7로 설정
chat = ChatOllama(model = "basic_kanana", temperature = 0.7)

# 사용자 입력이 있을 경우 (입력창에서 엔터를 눌렀을 때)
# st.chat_input() 함수에 입력 값이 있는 경우, prompt로 저장하고 조건절을 이어나감
if prompt := st.chat_input():
    # 입력된 사용자 메시지를 세션 상태에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 사용자 메시지를 채팅창에 출력
    st.chat_message("user").write(prompt)

    # AI 모델에 사용자 입력을 전달하여 응답 생성
    response = chat.invoke(prompt)
    msg = response.content  # 응답 결과에서 텍스트만 추출

    # AI 응답을 세션 상태에 저장
    st.session_state.messages.append({"role": "assistant", "content": msg})
    # AI 메시지를 채팅창에 출력
    st.chat_message("assistant").write(msg)