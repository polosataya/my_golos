import streamlit as st
import pandas as pd
import json
import os

# Установка переменной окружения KMP_DUPLICATE_LIB_OK в TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Мой голос",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None}
)

# Hide Streamlit logo
hide_streamlit_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load CSV data
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Load stop words
with open("model/stopwords.txt") as f:
    stopwords_nltk = f.read().splitlines()

# Upload JSON file
uploaded_file = st.file_uploader("Выберите файл JSON", type=["csv", "json"], label_visibility="collapsed")

# Load data from CSV file
data = load_data('data/labeled.csv')

# Get unique survey codes
codes = list(data['code'].unique())

# Select code from dropdown
code = st.selectbox('Выберите код опроса', codes)

# Add option to filter out typos, profanity, and numbers
orfo = st.sidebar.radio("Опечатки", ["оставить", "удалить"], key="orfo", help="Выберите, оставить или удалить опечатки")
obsc = st.sidebar.radio("Нецензурная лексика", ["оставить", "удалить"], key="obsc", help="Выберите, оставить или удалить нецензурную лексику")
tg = st.sidebar.radio("Числа", ["оставить", "удалить"], key="tg", help="Выберите, оставить или удалить числа")

temp = data[data.code == code].reset_index()

# Filter out typos, profanity, and numbers based on user selection
if orfo == "удалить":
    temp = temp[~temp['corrected'].str.contains(r'\b\w{1,2}\b')]
if obsc == "удалить":
    temp = temp[~temp['corrected'].str.contains(r'\b(?:мат|обсценную лексику|профанитет)\b', case=False, na=False)]
if tg == "удалить":
    temp = temp[~temp['corrected'].str.contains(r'\b\d+\b', na=False)]

st.write(temp['question'][0])

# Word Cloud visualization
text = temp['corrected'].str.cat(sep=' ')
wordCloud = WordCloud(width=400, height=300, random_state=1, background_color='white', colormap='coolwarm', max_words=100).generate(text)

# Set the color palette for the bar plot and pie chart
sns.set_palette("coolwarm")

# Use columns to layout elements
col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

# Leave col1 and col3 empty
with col1:
    pass

# Word Cloud, Cluster, and Sentiment in col2
with col2:
    st.subheader('Облако слов')
    fig1, ax1 = plt.subplots(figsize=(6, 4))  # Уменьшаем размеры графика
    ax1.imshow(wordCloud, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    st.subheader('Кластеры')
    temp2 = temp.groupby(by=["cluster"]).agg("sum")['count'].reset_index()
    temp2.sort_values(by=['count'], ascending=False, inplace=True)
    temp2 = temp2.head(10)  # Выбираем только первые 10 кластеров сверху
    fig2, ax2 = plt.subplots(figsize=(6, 4))  # Уменьшаем размеры графика
    sns.barplot(x='count', y='cluster', data=temp2)
    ax2.set_xlabel('Количество')
    ax2.set_ylabel('Кластер')
    ax2.set_title('Распределение по кластерам')
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader('Тональность ответов')
    temp1 = temp.groupby(by=["sentiment"]).agg("sum")['count'].reset_index()
    fig3, ax3 = plt.subplots(figsize=(6, 4))  # Уменьшаем размеры графика
    ax3.pie(temp1['count'], labels=temp1['sentiment'], autopct='%1.1f%%')
    ax3.set_title('Распределение тональности')
    st.pyplot(fig3)

# Leave col3 empty
with col3:
    pass

st.download_button(label="Download data as csv", data=temp.to_csv(index=False).encode("utf-8"), file_name='result.csv', mime='text/csv')
