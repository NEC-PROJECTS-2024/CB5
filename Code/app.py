from flask import Flask, render_template, request, jsonify, flash
import os
import librosa
import numpy as np
import joblib
from music_data_generator import extract_features 
from keras.models import load_model
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime  


app = Flask(__name__)
app.secret_key = '123456789'


#DATABASE
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://Ravi\\SQLEXPRESS/music_contact_us?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
db = SQLAlchemy(app)
# Define the Contact model
class Contact(db.Model):
    __tablename__ = 'contacts'  # Specify the correct table name

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)




# Load the saved CNN model
model = load_model('cnn_model.h5')
# Load the saved LabelEncoder and ColumnTransformer
label_encoder = joblib.load('label_encoder.pkl')
column_transformer = joblib.load('column_transformer.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'wav'



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Load the file directly into memory using librosa
            y, sr = librosa.load(file, sr=None)

            # Extract features
            features = extract_features(y, sr)

            # Apply the saved ColumnTransformer
            transformed_features = column_transformer.transform(features)

            # Make the prediction using the loaded model
            predicted_class = model.predict(transformed_features)          

            y_pred = tf.argmax(predicted_class, axis=1).numpy()
            class_label = label_encoder.inverse_transform(y_pred)

            # Display the predicted class on the webpage
            prediction = (class_label[0]).capitalize()

    return render_template('upload.html', prediction=prediction)



@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact_us.html')


@app.route('/submit_form', methods=['POST'])
def submit_form():
    flash('Form submitted successfully!', 'success')

    data = {
        'name': request.form['name'],
        'email': request.form['email'],
        'message': request.form['message']
    }

    print("Received data:", data)
    new_contact = Contact(name=data['name'], email=data['email'], message=data['message'])
    db.session.add(new_contact)
    db.session.commit()

    return render_template('contact_us.html')



@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')



@app.route('/help', methods=['GET', 'POST'])
def help():
    return render_template('help.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
