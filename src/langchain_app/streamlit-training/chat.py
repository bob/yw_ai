
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from langchain_app.model_config import DEFAULT_CHAT_MODEL, chat_model_options
load_dotenv()

SYSTEM_PROMPT = """Ты веселый шутник, массовик-затейник. Что ни слово, то искрометный юмор.

"""

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

model = init_chat_model(
    DEFAULT_CHAT_MODEL,
    **chat_model_options(DEFAULT_CHAT_MODEL), # type: ignore
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    streaming=True,
) # type: ignore

agent = create_agent(
    model=model,
    tools=[],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=st.session_state.checkpointer,
)

st.title("ChatGPT chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def v3_text_stream():
            stream = agent.stream_events(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
                version="v3"
            )

            for message in stream.messages:
                yield from message.text


        response = st.write_stream(v3_text_stream())
        st.session_state.messages.append({"role": "assistant", "content": response})
