from operator import ge
from types import MethodType
from flask import Flask, render_template, request, redirect
from flask.helpers import url_for
from flask.wrappers import Request
import generate_caption
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/upload_images/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template("index.html")



@app.route('/process', methods=["POST"])
def process():
    if request.method == "POST":
        if 'photo' not in request.files:
            return render_template("process.html", st='')
        photo = request.files['photo']
        path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
        photo.save(path)
    prediction = generate_caption.generate(path)
    # prediction="aa"
    return render_template("process.html", photo=url_for('static', filename='upload_images/'+photo.filename),  st=prediction)

if __name__=="__main__":
    app.run(debug=True)

