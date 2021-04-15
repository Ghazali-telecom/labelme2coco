from flask import Flask
from src.utils import random_number

app = Flask(__name__)


@app.route('/')
def hello():
    p = random_number(9)
    return f'Hello,World {p}'


@app.route('/simo')
def hello_simo():
    p = random_number(9)
    return f'Hello,World Simo  {p}'


if __name__ == '__main__':
    app.run(host='0.0.0.0')