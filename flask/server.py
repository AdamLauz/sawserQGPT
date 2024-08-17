from flask import Flask, request, jsonify
from sawser_q_gpt_service import sawserq_gpt_service

app = Flask(__name__)

# Initialize sawserq_gpt_service when the server starts
sawserq_gpt = sawserq_gpt_service()


@app.route("/query", methods=["POST"])
def predict():
    # get the user query
    user_query = request.get_json().get("query")

    # make a query
    answer = "dummy answer" #sawserq_gpt.query(user_query)

    # send back the answer in json format
    data = {"answer": answer}
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
