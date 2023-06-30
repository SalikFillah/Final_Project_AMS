# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:10:26 2023

@author: salik
"""

# Import Module
import dash
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# WordCloud Module
import wget
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/SalikFillah/Final_Project_AMS/main/data.csv')
df_neg = pd.read_csv('https://raw.githubusercontent.com/SalikFillah/Final_Project_AMS/main/bigram_negative.csv')
df_pos = pd.read_csv('https://raw.githubusercontent.com/SalikFillah/Final_Project_AMS/main/bigram_positive.csv')

url = "https://github.com/SalikFillah/Sentiment-Analysis/raw/main/Comfortaa-Bold.ttf"
filename = wget.download(url)

def barplot_data(dataframe):

    sentiment_counts = dataframe.value_counts()
    values_sample = sentiment_counts.keys().tolist()
    counts_sample = sentiment_counts.tolist()

    return values_sample, counts_sample

# Create Dash App
app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div(children=[
        html.H1(children=['Dashboard Sentiment Analysis'],
                style = {'textAlign': 'center'}),
        html.H2(children=['K Nearest Neighbor'],
                style = {'textAlign': 'center'})
        ]),
    html.Div(children=[
        html.Label('Pilih Kategori Sentimen'),
        dcc.Dropdown(id='dropdown',
                     options=[{'label': i, 'value': i} for i in df.Sentiment.unique()],
                     value = 'negative',
                     placeholder='Select...',
                     style = {'width': '500px'})
        ]),
    html.Div(children=[
        html.H3(children=['Bar Plot'],
                style = {'textAlign': 'center'}),
        html.Center('Bar Plot Banyak Sentimen'),
        dcc.Graph(id='barplot',
                  style = {'width': '800px', 'height': '800px', 'margin': '0 auto'})
        ]),
    html.Div([
        html.H3(children=['WordCloud'],
                style = {'textAlign': 'center'}),
        html.Center('WordCloud 100 Kata Terbanyak'),
        html.Br(),
        html.Br(),
        html.Div([html.Img(id= 'matplotlib-graph', className="img-responsive", style={'max-height': '520px', 'margin': '0 auto'})
            ], style={'margin': 'auto', 'width': '50%'})
            ]),
    html.Div([
        html.H3(children=['Bigram'],
                style = {'textAlign' : 'center'}),
        html.Center('Bigram 12 Pasangan Kata Terbanyak'),
        dcc.Graph(id='ngram',
                  style = {'width': '1000px', 'height': '1000px', 'margin': '0 auto'})
        ])
    ])

#-------------------------------------------------------------------------
@app.callback(Output('barplot', 'figure'),
              Input('dropdown', 'value'))

def update_barplot(value):
    
    df_bar = df[df['Sentiment']==value]
    values_sample, counts_sample = barplot_data(df_bar['Sentiment'])
    
    fig = go.Figure(data=[go.Bar(
        x=values_sample,
        y=counts_sample
        )])
    return fig
#-------------------------------------------------------------------------
@app.callback(
    Output('matplotlib-graph', 'src'),
    Input('dropdown', 'value'))

def update_wc(value):
    
    df_wc = df[df['Sentiment']==value]
    count = df_wc.clean_text.str.split(expand=True).stack().value_counts()[:100]
    angka = [0, 100]
    word = count[angka[0]:angka[1]]
    wordcloud = WordCloud(background_color = "white",
                          font_path=filename,
                          width = 600,
                          height = 400,
                          max_words = 100,
                          colormap = 'ocean').generate_from_frequencies(word)
    buf = io.BytesIO()
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buf, format = "png", dpi=600, bbox_inches='tight', pad_inches=0)
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
      
    return "data:image/png;base64,{}".format(data)
#-------------------------------------------------------------------------- 
@app.callback(
    Output('ngram', 'figure'),
    Input('dropdown', 'value'))

def create_plot(value):
    
    if value=='negative':
        
        fig = go.Figure(data=[go.Bar(
        x=df_neg['Bigram'],
        y=df_neg['Count'],
        marker=dict(color='blue')
)])
        return fig
    else :
        fig = go.Figure(data=[go.Bar(
        x=df_pos['Bigram'],
        y=df_pos['Count'],
        marker=dict(color='blue')
)])     
        return fig


if __name__ == '__main__':
    app.run_server(debug=True)
