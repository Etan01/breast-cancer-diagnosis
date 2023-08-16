from distutils.log import debug
from email import message
import os
from pickle import TRUE
from pyexpat.errors import messages
import flask
from flask import Flask, redirect, url_for, render_template, request, session, flash
from werkzeug.utils import secure_filename
import tensorflow as tf

#Keras Package
from keras.models import model_from_json
from tensorflow.keras.applications.vgg19 import VGG19 # VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Softmax

import numpy as np
import cv2
from PIL import Image

from authentication import config
import pyrebase

UPLOAD_FOLDER = "static/upload/"

#initialise Flask
app = flask.Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Firebase configuration
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#Initialze person as dictionary
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}

#function: Get VGG19 pre-trained model and add layers
def get_model():
    base_model = VGG19(input_shape=(224,224,3), weights='imagenet', include_top=False)
    model= Sequential()
    model.add(base_model)
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    return model

model = get_model()
model.load_weights('models/breast_cance_vgg3_weights-4.h5')
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#function: Convert image to array
def get_img_data(image):
    img = cv2.imread(image, 1)
    img = cv2.resize(img, (224, 224))
    rows, cols, color = img.shape
    total_angle = 360
    data = {}
    for angle in range(0, total_angle, 8):
        matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1) 
        img_rotated = cv2.warpAffine(img, matrix, (cols, rows))
        data[angle] = img_rotated

    img_data = []
    img_data += data.values()
    img_data = np.array(img_data)
    return img_data

#function: prediction
def predict(image_bytes):
    img_data = get_img_data(image_bytes)
    prediction = model.predict(img_data)[0]

    pred_class = (prediction> 0.5)
    percentage = float(prediction[pred_class])
    result = pred_class.astype("int32")
    pred_class = np.argmax(result,-1)

    pred = predict_result(pred_class)
    warning = None
    if percentage == 1:
        warning = 'Our system detects that your image is not medical images'
        output = 'Breast Cancer Diagnosis is {}'.format(pred)
    else:
        output = 'Breast Cancer Diagnosis is {}'.format(pred)

    return (output, warning)

#Classification
def predict_result(pred):
    if pred == 2:
        return 'Benign'
    elif pred == 1:
        return 'Malignant'
    else:
        return 'Normal'
    
#Login
@app.route("/")
def login():
    return render_template("login.html")

#Sign up/ Register
@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/login", methods = ['GET','POST'])
def login_successfully():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["pass"]
        print(email, password)
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            print("Login successful")
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            print(person)

            #Get the name of the user
            data = db.child("users").get()
            person["name"] = data.val()[person["uid"]]["name"]
            #Redirect to homepage
            print("Redirecting to homepage")
            return render_template("index.html")
        except Exception as e:
            print(e)
            #If there is any error, redirect back to login
            return render_template("login.html", messages = "Username or password is incorrect")
    print("Login unsuccessful")
    return render_template("login.html")

#If someone clicks on the register
@app.route("/register", methods = ['GET','POST'])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["pass"]
        name = request.form["name"]
        try:
            #Try creating the user account using the provided data
            auth.create_user_with_email_and_password(email, password)
            #Login the user
            user = auth.sign_in_with_email_and_password(email, password)
            #Add data to global person
            global person
            person["is_logged_in"] = True
            person["email"] = user["email"]
            person["uid"] = user["localId"]
            person["name"] = name
            #Append data to the firebase realtime database
            print("Adding data to firebase")
            data = {"name": name, "email": email}
            results = db.child("users").child(person["uid"]).set(data)
            #Go to welcome page
            return render_template("index.html")
        except Exception as e:
            print(e)
            #If there is any error, redirect to register
            return render_template("signup.html", messages = "Error creating account")
    else:
        if person["is_logged_in"] == True:
            return render_template("index.html")
        else:
            return redirect(url_for('register'))


#Homepage
@app.route('/homepage',  methods = ['GET','POST'])
def home():
    if person ["is_logged_in"] != True:
        return redirect(url_for('login'))

    error = None
    if request.method == "POST":
        if request.form["submit"] == "submit":
            if not request.files.get("file", None):
                error = "No image fill"
                return render_template("error.html", messages = error)
            file = request.files.get("file")

            if not file:
                return

            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            if file:
                filename = secure_filename(file.filename)
                full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                #flash('File successfully uploaded')
            try:
                output, warning = predict(full_filename) #call function predict
                if warning is not None:
                    return render_template("error.html", messages=warning)
            except:
                return render_template("error.html")
        return render_template("result.html", name=output, user_image = full_filename, warning=warning)
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    if person ["is_logged_in"] != True:
        return redirect(url_for('login'))
    return render_template("about_us.html")    

@app.route('/about_model')
def about_model():
    if person ["is_logged_in"] != True:
        return redirect(url_for('login'))
    return render_template("about_model.html")   

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for(filename='upload/' + filename), code=301)

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    
    app.secret_key = "secret"
    app.run(debug=True)



