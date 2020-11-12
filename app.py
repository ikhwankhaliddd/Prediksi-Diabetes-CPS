#Program prediksi penderita diabetes

#Import Library
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Buat judul dan sub-judul
st.write("""
# Prediksi Diabetes
Akan mendeteksi apakah seseorang memiliki diabetes atau tidak menggunakan Random Forest Classifier Algorithm dan Python!
""")

#Display Gambar
image = Image.open('Diabetes Prediction using machine learning.PNG')
st.image(image, caption = "Version 1.0", use_column_width=True)


#Masukkan Dataset
df = pd.read_csv('diabetes.csv')

#Set Subheader
st.subheader('Data Information:')

#Buat Data menjadi Tabel
st.dataframe (df)

#Buat statistik dari data
st.write(df.describe())

#Tampilkan data menjadi chart

chart = st.bar_chart(df)

#Bagi data menjadi 2 yaitu 'X' dan "Y"
X=df.iloc[:, 0:8].values
Y=df.iloc[:,-1].values

#Bagi dataset menjadi 75 % data training dan 25% data testing
X_train, X_test, Y_train, Y_test = train_test_split (X,Y, test_size=0.25, random_state = 0)

#Input Feature Dari User

def get_user_input() :
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness= st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0 , 30.5)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725) 
    age = st.sidebar.slider('Age', 21, 81, 29)

    #Store dictionary ke variable
    user_data = {'Pregnancies' : pregnancies,
                            'Glucose' : glucose,
                            'Blood Pressure' : blood_pressure,
                            'Skin Thickness' : skin_thickness,
                            'Insulin' : insulin,
                            'BMI' : bmi,
                            'Diabetes Pedigree Function' : dpf,
                            'Age' : age
                            }

    #Ubah data ke data frame
    features = pd.DataFrame (user_data, index = [0])
    return features

#Store Input User ke variabel
user_input = get_user_input()

#Set subheader dan tampilkan input user
st.subheader('User Input :' )
st.write(user_input)

#Buat dan Training Model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


#Tampilkan metrics model
st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100 ) + ' %')

#Store prediksi model ke variabel
prediction = RandomForestClassifier.predict(user_input)

#Set subheader dan tampilkan hasil klasifikasi

st.subheader( " Classification : ")
st.write(prediction)
