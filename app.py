from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import warnings

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
warnings.filterwarnings("ignore")

#Load your RF model here
loaded_modelRF = joblib.load(open('RF4Class.h5','rb'))

# Dictionary to track prediction status for each image
prediction_status = {}

# Render the HTML page
@app.route('/')
def index():
    return 'Welcome To My Application'


# Load and predict image function
def load_and_predict_image4(image_path):
    
    activation = 'relu'
    feature_extractor = Sequential()
    feature_extractor.add(Conv2D(32, 3, activation = activation, input_shape = (224, 224, 3)))
    feature_extractor.add(BatchNormalization())

    feature_extractor.add(Conv2D(32, 3, activation = activation))
    feature_extractor.add(BatchNormalization())
    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Conv2D(64, 3, activation = activation))
    feature_extractor.add(BatchNormalization())
    feature_extractor.add(MaxPooling2D())

    feature_extractor.add(Conv2D(128, 3, activation = activation))
    feature_extractor.add(BatchNormalization())
    feature_extractor.add(MaxPooling2D())
    feature_extractor.add(Dropout(0.25))

    feature_extractor.add(Flatten())

    # Create a LabelEncoder object
    Label_Encoder = LabelEncoder()
    Label_Encoder.fit(["Mild", "Moderate", "Severe", "Proliferate_DR"])  # Fit the encoder on the label classes
    labels_encoded = Label_Encoder.transform(["Mild", "Moderate", "Severe", "Proliferate_DR"])

    # Open the image using PIL
    img = Image.open(image_path)
    # Resize the input image
    input_img = img.resize((224, 224))
    input_img = np.array(input_img)
    input_img = input_img / 255.0
    
    # Expand dims so the input is (num images, x, y, c)
    input_img = np.expand_dims(input_img, axis=0)

    # Extract features from the input image using the feature extractor
    input_img_features = feature_extractor.predict(input_img)

    # Predict the category using the trained RF model
    prediction_RF_encoded = loaded_modelRF.predict(input_img_features)[0]
    
    # Reverse the label encoder to original name
    predicted_label = Label_Encoder.inverse_transform([prediction_RF_encoded])[0]
    
    # Return the prediction label
    return predicted_label


# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# API endpoint for image upload
@app.route('/predict', methods=['POST'])
def predict():
    global prediction_made
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    # Check if file has a valid extension
    if file and allowed_file(file.filename):
        # Check if prediction has already been made for this image
        if file.filename in prediction_status:
            predicted_label = "Prediction already made."
        else:
            # Call load_and_predict_image function
            predicted_label = load_and_predict_image4(file)
            # Mark prediction status as True for this image
            prediction_status[file.filename] = True

        # Return the predicted label
        return jsonify({'predicted_label': predicted_label})
    else:
        return jsonify({'error': 'Invalid file'})

if __name__ == '__main__':
    app.run()
