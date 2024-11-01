from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

resources = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_resource():
    title = request.form.get('resourceTitle')
    description = request.form.get('resourceDescription')
    url = request.form.get('resourceURL')

    resource = {
        'title': title,
        'description': description,
        'url': url
    }
    resources.append(resource)

    return jsonify({'message': 'Resource submitted successfully!'}), 201

if __name__ == '__main__':
    app.run(debug=True)
