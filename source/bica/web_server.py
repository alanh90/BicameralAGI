from flask import Flask, request, jsonify, render_template
from bica.gpt_handler import GPTHandler

app = Flask(__name__,
            static_folder='../web/static',
            template_folder='../web/templates')

gpt_handler = GPTHandler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']

    # Generate response
    response = gpt_handler.generate_response([
        {"role": "system", "content": "You are BICA, a helpful AI assistant. Provide a concise response."},
        {"role": "user", "content": user_input}
    ])

    return jsonify({
        "response": response
    })

if __name__ == '__main__':
    app.run(debug=True)