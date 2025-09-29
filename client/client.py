import requests

URL = "http://<IP>/query"
TEST_QUERY = "What is the most notable fact about the Circassians?" #"Who are the Circassians? where did they come from?"

if __name__ == "__main__":
    # Send the query string in the POST request
    response = requests.post(URL, json={'query': TEST_QUERY}, stream=True)
    answer = response.content.decode()

    print(f"Answer from Sawser Q GPT is: {answer}")