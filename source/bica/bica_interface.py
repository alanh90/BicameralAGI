import json
from flask import Flask, request, jsonify
from bica_orchestrator import BicaOrchestrator
from bica_logging import BicaLogging

app = Flask(__name__)
logger = BicaLogging("BicaInterface")


class BicaInterface:
    def __init__(self):
        self.orchestrator = BicaOrchestrator()

    def process_request(self, request_data):
        # Forward the request to the orchestrator and return its response
        response = self.orchestrator.process_input(request_data)
        logger.info("Processed request", input=request_data, response=response)
        return response


bica_interface = BicaInterface()


@app.route('/interact', methods=['POST'])
def interact():
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        response = bica_interface.process_request(data)
        return jsonify(response), 200
    except Exception as e:
        logger.error("Error processing request", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    try:
        status = bica_interface.orchestrator.get_system_status()
        return jsonify(status), 200
    except Exception as e:
        logger.error("Error getting system status", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
