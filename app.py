# serve.py
from flask import Flask, request, render_template, send_from_directory, Blueprint, jsonify, Response
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
import logging,os,subprocess,sys
from script import ClouderizerEval
from functools import wraps
import json
import threading
import logging

# creates a Flask application, named app
app = Flask(__name__)
cldz_eval = ClouderizerEval(app)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
OUTPUT_FOLDER = '{}/output/'.format(PROJECT_HOME)
app.config.from_json('config.json')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
#app.config['COMMAND'] = "python ../code/fast_neural_style/neural_style/neural_style.py eval --content-image $IMG1$ --model ../code/fast_neural_style/saved_models/rain_princess.pth --output-image $OIMG1$ --cuda 0"
#app.config['COMMAND'] = ""

sys.path.append(os.path.abspath("../code/"))
from fastai.torch_imports import *
from fastai.core import *
from fasterai.filters import Colorizer34
from fasterai.visualize import ModelImageVisualizer


def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == app.config['UNAME'] and password == app.config['PASSWORD']

def allowed_file(filename):
    return '.' in filename and \
      filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required. Default credentials admin/admin"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def updateConfig(key, value):
    jsonFile = open("config.json", "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Working with buffered content
    data[key] = value
    
    ## Save our changes to JSON file
    jsonFile = open("config.json", "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

# Quick fix for 504 Gateway error due to model initialization

colorizer_path = '../code/colorize_gen_192.h5'
render_factor = 42

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

cldz_serve_outputdir = "./output"

filters=None
vis=None

def run_job():
    global filters, vis
    print('Loading models')
    filters = [Colorizer34(gpu=0, weights_path=colorizer_path,nf_factor=2, map_to_orig=True)]
    vis = ModelImageVisualizer(filters, render_factor=42, results_dir=cldz_serve_outputdir)
    print('model loading is done')
    
modelThread = threading.Thread(target=run_job)
modelThread.start()
    
    
def model_inference(requestparams):
    if 'image' not in requestparams:
      return BadRequest("File not present in request")
    file = requestparams['image']
    if file.filename == '':
      return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
      return BadRequest("Invalid file type")
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      output_filepath = Path(os.path.join(app.config['OUTPUT_FOLDER'], filename))
      file.save(input_filepath)

#     modelThread.join()
    print(vis)
    if vis is not None:
        try:    
            result = vis.get_transformed_image_ndarray(input_filepath)
            vis.save_result_image(output_filepath, result)
        except AttributeError:
            pass
    else:
        return jsonify("Model initialising"), "object"
    output = {}
    output["directory"]=app.config['OUTPUT_FOLDER']
    output["filename"]=filename
#     app.logger.info(requestparams)
    return output, "imagepath"

# a route where we will display a welcome message via an HTML template
@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def home(path):  
    dir = os.path.join(os.curdir, 'static')
    bp = Blueprint('cast', __name__, static_folder=dir, static_url_path='')
    if(path == ''):
        return bp.send_static_file('index.html')
    else:
        return bp.send_static_file(path)

@app.route("/api/eval", methods = ['POST'])
def eval(): 
    output, outputtype = model_inference(CombinedMultiDict((request.files, request.form)))
    if outputtype == 'imagepath':
        return send_from_directory(directory=output['directory'],filename=output['filename'], as_attachment=True)
    elif outputtype == 'object':
        return output
    else:
        return "Some error occured"


@app.route("/api/script", methods = ['GET'])
@requires_auth
def getEvalCode():
    with open("script.py", "r") as f:
        content = f.read()
    return jsonify(
        text=content
    )

@app.route("/api/script", methods = ['POST'])
@requires_auth
def updateEvalCode():
    script = request.form['script']
    with open("script.py", "w") as f:
        f.write(script)
    return jsonify(
        success=True
    )

@app.route("/api/command", methods = ['GET'])
@requires_auth
def getCommand():
    return jsonify(
        text=app.config['COMMAND']
    )

@app.route("/api/command", methods = ['POST'])
@requires_auth
def updateCommand():
    command = request.form['command']
    app.config['COMMAND'] = command
    updateConfig('COMMAND', command)
    return jsonify(
        success=True
    )

@app.route("/api/projname", methods = ['GET'])
def getProjName():
    return jsonify(
        projectname=app.config['PROJECTNAME']
    )

@app.route("/api/credentials", methods = ['POST'])
@requires_auth
def updateCredentials():
    uname = request.form['uname']
    password = request.form['password']
    app.config['UNAME'] = uname
    app.config['PASSWORD'] = password
    updateConfig('UNAME', uname)
    updateConfig('PASSWORD', password)
    return Response(
    'Credentials updated. Login again.', 401)

@app.route("/api/logout", methods = ['GET'])
@requires_auth
def logout():
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401)


# run the application
if __name__ == "__main__":  
    app.run(debug=True)
