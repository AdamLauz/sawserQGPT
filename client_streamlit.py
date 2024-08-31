import requests

URL = "http://13.48.10.35/query"
TEST_QUERY = "What is the most notable fact about the Circassians?" #"Who are the Circassians? where did they come from?"

# if __name__ == "__main__":
#     # Send the query string in the POST request
#     response = requests.post(URL, json={'query': TEST_QUERY})
#     data = response.json()
#
#     print(f"Answer from Sawser Q GPT is: {data['answer']}")
#


import streamlit as st

# Show title and description.
st.title("💬 SawserQ GPT Chatbot")
st.write(
    "This is a simple chatbot that uses a model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
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
    response = requests.post(URL, json={'query': prompt}, stream=True)


    # Function to yield chunks of text
    def generate_chunks(response):
        for chunk in response.iter_content(chunk_size=128):
            if chunk:
                yield chunk.decode('utf-8')


    # Initialize Streamlit for displaying the response
    with st.chat_message("assistant"):
        full_response = ""
        for chunk in generate_chunks(response):
            full_response += chunk

    # Display the entire response
    #st.write("Full Response:", full_response)

    # Extract and display text after the delimiter [/INST]
    delimiter = "[/INST]"
    if delimiter in full_response:
        response_part = full_response.split(delimiter, 1)[1].strip()  # Extract text after the delimiter
        st.write(response_part)
    else:
        st.write("Delimiter not found in response.")

