import torch.multiprocessing as mp
from flask import Flask, request, jsonify
from sawser_q_gpt_service import sawserq_gpt_service


app = Flask(__name__)

# Set start method for multiprocessing
mp.set_start_method('spawn', force=True)

# Initialize sawserq_gpt_service when the server starts
sawserq_gpt = sawserq_gpt_service()


@app.route("/query", methods=["POST"])
def predict():
    try:
        # Set start method for multiprocessing
        mp.set_start_method('spawn', force=True)

        # get the user query
        user_query = request.get_json().get("query")

        # make a query
        answer = sawserq_gpt.query(user_query)

        # send back the answer in json format
        data = {"answer": answer}
        return jsonify(data)

    except Exception as e:
        # Return an error response in case of failure
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
