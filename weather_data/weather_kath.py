import streamlit as st 
import joblib 
import pandas as pd
# laod the model 
model = joblib.load(r"D:\New Folder\weather_data\model_weather.pt")
# function to encode and predcit data 
def predict(data): 
    data=data.dropna()
    df=data.copy()
    # encode 
    enc_season={'Winter':1, 'Spring':2, 'Autumn':3}
    df['Season']=data['Season'].map(enc_season)
    # after encode 
    data['predicted']=model.predict(df)

    return data 

# title of applciation 
st.title("Weather Prediction")
file=st.file_uploader("Upload Your file", type='csv')
try:
    if file is not None:
        data=pd.read_csv(file)
        st.write("first five rosw")
        st.write(data.head())
        df_p=predict(data)
        st.write("Data with prediction")
        st.write(df_p)
    else:
        st.write("Empty file cannot be read")
except Exception as e: 
    st.write(f"Error {e} occured")

finally: 
    st.write("Thank you for using our service ")