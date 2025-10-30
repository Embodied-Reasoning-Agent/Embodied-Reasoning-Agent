import argparse
import time
from flask import Flask, request, jsonify

# Import your AguvisModel from wherever it's defined.
# from my_model_file import AguvisModel
# For this example, we'll assume it's in "my_model_module.py".
from embodiedbench.planner.aguvis_model import AguvisModel

app = Flask(__name__)

# We'll keep a global reference to the loaded model so it's accessible in the /respond endpoint.
model = None

@app.route("/respond", methods=["POST"])
def respond():
    """
    Expects a JSON payload of the form:
    {
      "prompt": "...",
      "obs": "...",
      "previous_actions": "...",
      "low_level_instruction": "...",
      "mode": "self-plan",
      "temperature": 0.0,
      "max_new_tokens": 1024
    }

    Returns:
    {
      "response": "Model-generated text"
    }
    """

    data = request.get_json(force=True) or {}

    messages = data.get("message", [])
    mode = data.get("mode", "self-plan")
    temperature = data.get("temperature", 0.01)
    max_new_tokens = data.get("max_new_tokens", 1024)
    print(f"config: mode = {mode}, temperature = {temperature}, max_new_tokens = {max_new_tokens}")

    try:
        response_text = model.respond(
            messages = messages,
            mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    except Exception as e:
        # Optionally handle or log the error
        print("An unexpected error occurred:", e)
        # Retry once, similar to your original logic
        time.sleep(20)
        response_text = model.respond(
            messages = messages,
            mode=mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

    return jsonify({"response": response_text})

def main():
    parser = argparse.ArgumentParser(description="Aguvis Model Server")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="MODEL_PATH", # change this 
        help="Path to the ERA (or any other) model."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", # change this if needed
        help="Device for running the model (e.g. 'cuda' or 'cpu')."
    )

    args = parser.parse_args()

    # Load the model once at server startup.
    global model
    model = AguvisModel(model_path=args.model_path, device=args.device)

    # Run the Flask app. Adjust host/port if needed.
    app.run(host="0.0.0.0", port=5001) # change this if needed

if __name__ == "__main__":
    main()