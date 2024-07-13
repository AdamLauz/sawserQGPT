from flask import Flask, request, jsonify
from sawser_q_gpt_service import sawserq_gpt_service

app = Flask(__name__)


@app.route("/query", methods=["POST"])
def predict():
    # get the user query
    user_query = request.get_json().get("query")

    # invoke sawser Q GPT service
    sawserq_gpt = sawserq_gpt_service()

    # make a query
    answer = sawserq_gpt.query(user_query)

    # send back the answer in json format
    data = {"answer": answer}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)