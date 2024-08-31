import requests

URL = "http://<IP>/query"


import streamlit as st

# Show title and description.
st.title("💬 SawserQ GPT Chatbot")
st.write(
    "This is a chatbot that uses a SawserQ GPT model to generate responses. "
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the API with streaming enabled
    answer = requests.post(URL, json={'query': prompt}).content.decode()

    # Stream the response to the chat using `st.write_stream`, then store it in
    # session state.
    with st.chat_message("assistant"):
        st.write(answer.split("</s>")[0])
    st.session_state.messages.append({"role": "assistant", "content": answer})




