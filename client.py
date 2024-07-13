import requests

URL = "http://127.0.0.1/query"
TEST_QUERY = "Who are the Circassians? where did they come from?"

if __name__ == "__main__":
    # Send the query string in the POST request
    response = requests.post(URL, json={'query': TEST_QUERY})
    data = response.json()

    print(f"Answer from Sawser Q GPT is: {data['answer']}")