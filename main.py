import streamlit as st
from background import set_background

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Brain", "Breast", "Skin", "AIDS", "Lung"])

if page == "Home":
    st.title('Welcome')

elif page == "Lung":
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    import numpy as np
    from PIL import Image

    # Load the saved model
    model = load_model('model/lung_cancer_inceptionv3.h5')

    # Define class labels (in the same order used for training)
    class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']


    # Function to preprocess uploaded images for prediction
    def preprocess_image(image, target_size=(299, 299)):
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Rescale as during training
        return image


    # Streamlit app UI
    st.title("Lung Cancer Prediction")
    st.write("Upload a chest X-ray image to predict the type of lung cancer.")

    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Display prediction result
        st.write(f"Prediction: **{class_labels[predicted_class]}**")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")



elif page == "AIDS":
    st.title('AIDS Prediction')
    import numpy as np
    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from imblearn.over_sampling import RandomOverSampler

    df1 = pd.read_csv("model/AIDS_Classification.csv")
    X = df1.drop('infected', axis=1)
    y = df1['infected']
    oversampler = RandomOverSampler(random_state=42)
    X_oversampled, y_oversampled = oversampler.fit_resample(X, y)
    pre_process = preprocessing.MinMaxScaler().fit(X_oversampled)
    x_transform = pre_process.fit_transform(X_oversampled)
    x_train, x_test, y_train, y_test = train_test_split(x_transform, y_oversampled, test_size=.10, random_state=101)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    # time = 1069
    # trt = 2
    # age = 46
    # wtkg = 63.8
    # hemo = 0
    # homo = 1
    # drugs = 0
    # karnof = 100
    # oprior = 0
    # z30 = 1
    # preanti = 881
    # race = 0
    # gender = 1
    # str2 = 1
    # start = 3
    # symptom = 0
    # treat = 1
    # offtrt = 1
    # cd40 = 330
    # cd420 = 320
    # cd80 = 820
    # cd820 = 630

    # Collect data from the user using Streamlit widgets
    time = st.number_input("Enter time:", min_value=0, format='%d')
    trt = st.number_input("Enter trt:", min_value=0, format='%d')
    age = st.number_input("Enter age:", min_value=0, format='%d')
    wtkg = st.number_input("Enter wtkg:", min_value=0.0, format='%f')
    hemo = st.number_input("Enter hemo:", min_value=0, format='%d')
    homo = st.number_input("Enter homo:", min_value=0, format='%d')
    drugs = st.number_input("Enter drugs:", min_value=0, format='%d')
    karnof = st.number_input("Enter karnof:", min_value=0, format='%d')
    oprior = st.number_input("Enter oprior:", min_value=0, format='%d')
    z30 = st.number_input("Enter z30:", min_value=0, format='%d')
    preanti = st.number_input("Enter preanti:", min_value=0, format='%d')
    race = st.number_input("Enter race:", min_value=0.0, format='%f')
    gender = st.number_input("Enter gender:", min_value=0, format='%d')
    str2 = st.number_input("Enter str2:", min_value=0, format='%d')
    start = st.number_input("Enter start:", min_value=0, format='%d')
    symptom = st.number_input("Enter symptom:", min_value=0, format='%d')
    treat = st.number_input("Enter treat:", min_value=0, format='%d')
    offtrt = st.number_input("Enter offtrt:", min_value=0, format='%d')
    cd40 = st.number_input("Enter CD40 count:", min_value=0, format='%d')
    cd420 = st.number_input("Enter cd420:", min_value=0, format='%d')
    cd80 = st.number_input("Enter CD80 count:", min_value=0, format='%d')
    cd820 = st.number_input("Enter cd820:", min_value=0, format='%d')

    user_data = np.array([[time, trt, age, wtkg, hemo, homo, drugs, karnof, oprior, z30, preanti, race, gender, str2,
                           start, symptom, treat, offtrt, cd40, cd420, cd80, cd820]])
    st.write(user_data)

    # Reshape the user input to match the expected input shape of the scaler
    user_data_reshaped = np.array(user_data).reshape(1, -1)

    # Transform the user input using the fitted scaler
    user_data_scaled = pre_process.transform(user_data_reshaped)


    # Make prediction using the loaded model
    if st.button('Predict'):
        try:
            y_pred_rfc = rfc.predict(user_data_scaled)

            # Display the prediction
            if y_pred_rfc[0] == 1:
                st.write("The model predicts that the person is infected with AIDS.")
            else:
                st.write("The model predicts that the person is not infected with AIDS.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

elif page == "Brain":
    set_background('./bgs/bg5.png')
    st.title('Brain Tumor Classification')
    st.header('Please upload image')

    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    if file is not None:
        from PIL import Image
        from keras.models import load_model

        # Load classifier
        model = load_model('model/brain_model.h5')

        # Load class names
        with open('model/brain_labels.txt', 'r') as f:
            class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

        # Display image
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # Classify image
        from brain_util import classify
        class_name, conf_score = classify(image, model, class_names)

        # Write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))


elif page == "Breast":
    import streamlit as st
    from keras.models import load_model
    from PIL import Image
    from util import classify


    set_background('./bgs/bg5.png')
    st.title('Breast Cancer classification')
    st.header('Please upload image')
    file = st.file_uploader('Upload your image here', type=['jpeg', 'jpg', 'png'], label_visibility='hidden')
    try:
        model = load_model('model/breast_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # load class names
    try:
        with open('model/breast_labels.txt', 'r') as f:
            class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
            f.close()
        # st.success("Class names loaded successfully")
    except Exception as e:
        st.error(f"Error loading class names: {e}")

    # display image
    if file is not None:
        try:
            image = Image.open(file).convert('RGB')
            st.image(image, use_column_width=True)

            # classify image
            class_name, conf_score = classify(image, model, class_names)

            # write classification
            st.write("## {}".format(class_name))
            st.write("### score: {}%".format(int(conf_score * 1000) / 10))
        except Exception as e:
            st.error(f"Error during classification: {e}")


elif page == "Skin":
    import streamlit as st
    from keras.models import load_model
    from PIL import Image
    from skin_util import classify


    set_background('./bgs/bg5.png')
    st.title('Skin Cancer classification')
    st.header('Please upload image')
    file = st.file_uploader('Upload your image here', type=['jpeg', 'jpg', 'png'], label_visibility='hidden')
    try:
        model = load_model('model/skin_cancer_model.h5')
        st.write("Model loaded successfully!")
    except Exception as e:
        st.write(f"Error loading model: {e}")

    # load class names
    try:
        with open('model/skin_labels.txt', 'r') as f:
            class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
            f.close()
        # st.success("Class names loaded successfully")
    except Exception as e:
        st.error(f"Error loading class names: {e}")

    # display image
    if file is not None:
        try:
            image = Image.open(file).convert('RGB')
            st.image(image, use_column_width=True)

            # classify image
            class_name, conf_score = classify(image, model, class_names)

            # write classification
            st.write("## {}".format(class_name))
            st.write("### score: {}%".format(int(conf_score * 1000) / 10))
        except Exception as e:
            st.error(f"Error during classification: {e}")
