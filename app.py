import streamlit as st

from humana_take_home.agents.research import ResearchAgent

st.set_page_config(
    page_title="Humana Take Home AI Assignment",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Humana Take Home AI Assignment")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the trained materials"}
    ]

research_agent = ResearchAgent.from_local_storage()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = research_agent.get_chat_engine()


if prompt := st.chat_input("Ask Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
