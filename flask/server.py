import torch.multiprocessing as mp
from flask import Flask, request, jsonify, Response
from sawser_q_gpt_service import sawserq_gpt_service
import time

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

        # generate streaming
        def generate_stream():
            try:
                # Generate the streaming response
                for token in sawserq_gpt.query(user_query):
                    yield token
            except Exception as e:
                # Handle exceptions in the generator
                yield f"Error: {str(e)}"

        # Return streaming response
        return Response(generate_stream(), content_type='text/plain')

    except Exception as err:
        # Return an error response in case of failure
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(debug=False)
