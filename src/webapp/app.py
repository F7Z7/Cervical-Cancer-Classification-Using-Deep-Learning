from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "<p>Hello, World!</p>"

@app.route('/hello/<name>')
def welcome(name):
    return 'hello %s' %name

if __name__ == '__main__':
    app.run(debug=True)