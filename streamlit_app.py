import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Engagement Score UI", layout="wide")
st.title("Engagement Score Predictor - Streamlit Interface")

DATA_PATH = "C:\\Users\\susmitha\\Downloads\\engagement.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def train_model(df):
    x = df[["interest_level", "sleep_hours"]]
    y = df["engagement_score"]
    model = LinearRegression().fit(x, y)
    return model, x, y

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find CSV at {DATA_PATH}")
    st.stop()

with st.expander("Data preview"):
    st.dataframe(df.head(10))

model, x_train, y_train = train_model(df)

st.sidebar.header("User input")
interest_level = st.sidebar.slider(
    "Interest Level",
    float(df["interest_level"].min()),
    float(df["interest_level"].max()),
    float(df["interest_level"].median()),
    step=0.1,
)
sleep_hours = st.sidebar.slider(
    "Sleep Hours",
    float(df["sleep_hours"].min()),
    float(df["sleep_hours"].max()),
    float(df["sleep_hours"].median()),
    step=0.1,
)

feature_choice = st.sidebar.selectbox(
    "Feature set",
    ["interest_level + sleep_hours"]
)

if st.sidebar.button("Predict"):
    value = np.array([[interest_level, sleep_hours]])
    prediction = model.predict(value)[0]
    st.success(f"Predicted Engagement Score: {prediction:.3f}")
    st.write("Coefficient:", model.coef_)
    st.write("Intercept:", model.intercept_)

st.markdown("---")

st.write("### Training data sample predictions")
pred_sample = model.predict(x_train)
st.dataframe(pd.concat([y_train.reset_index(drop=True), pd.Series(pred_sample, name="predicted")], axis=1).head(15))
