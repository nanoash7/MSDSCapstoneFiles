from dash import Dash, html, dcc, callback, Output, Input,ctx
import plotly.express as px
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go

from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from datetime import date


import pandas as pd
from datetime import datetime, timedelta,time
import glob
import json
import numpy as np

import sys
import os
sys.path.append(os.path.abspath('..'))

# Default color scales
colorscales = px.colors.named_colorscales()

default_color_scale = ["black", "black", "blue", "blue", "green", "yellow", "orange"]
colorscales.append('default_color_scale')


from data_processing.B_FFT_Graph_Generator.FFT_Generator_dashboard import main as fft_generator

with open('config.json', 'r') as f:
    config = json.load(f)

file_folder = config['file_folder']

# Sort the csv files in order.
csv_files = glob.glob(f'{file_folder}*.csv')

csv_files.sort()
csv_file_df = pd.DataFrame(csv_files,columns=['file_name'])

date_format = "%Y-%m-%dT%H-%M-%S"

# Adds the date as a column to the csv dataframe.
csv_file_df['date'] = csv_file_df['file_name'].apply(lambda x: datetime.strptime('-'.join(x.split("data-")[-1].split('-')[:-1]), date_format))
csv_file_df['date'] = pd.to_datetime(csv_file_df['date'])
csv_file_df = csv_file_df.sort_values('date')


image_folder = config['image_folder']
first_file = config.get('default_file',glob.glob(file_folder + '*')[0])


def get_files(start_time,end_time) : 
    """
    Helper function that gets the parquet files from the start to end timeframe. 
    
    """
    if start_time is not None : 
        csv_files_return = []
        got_start = False
        csv_between_df = csv_file_df[(csv_file_df['date']+ timedelta(seconds=10) > start_time) &(csv_file_df['date']< end_time + timedelta(seconds=10))]

        return csv_between_df['file_name'].values
    else : 
        return [csv_files[0]]

def get_df_round(start_time,end_time) :
    """
    Helper function that gets the necessary dataframe from the parquet file folder based on the start and end time frame needed.
    
    """
    if start_time is not None : 
        if end_time is None :
            multiplier = start_time
            first_time = pd.Timestamp(csv_file_df['date'].values[0])
        
            if multiplier != 0 : 
                start_time = first_time + timedelta(seconds=multiplier * 10)
            else :
                start_time = first_time 
            end_time = first_time + timedelta(seconds=(multiplier + 1) * 10)

        df = pd.DataFrame() 

        got_start = False
        for ind,x in csv_file_df.iterrows() :
            if not got_start : 
                if x['date'] + timedelta(seconds=10) > start_time : 
                    df = pd.concat([df,pd.read_csv(x['file_name'])])
                    got_start = True
            else : 
                if x['date'] < end_time + timedelta(seconds=10) : 
                    df = pd.concat([df,pd.read_csv(x['file_name'])])
        
    else : 
        df = pd.read_csv(first_file)
        
        
    date_string =   '-'.join(csv_files[0].split("data-")[-1].split('-')[:-1])#csv_files[0].split('/')
  
    date_format = "%Y-%m-%dT%H-%M-%S"

    date_object = datetime.strptime(date_string, date_format)

    df_round = df
    df_round = df_round.rename(columns={'Channel 1':'Channel 2','Channel 0':'Channel 1'})
    df_round['Time'] = df_round['Time'].astype(float) - 4791
    df_round['Time'] = df_round['Time'].round(3)
    df_round['Datetime'] = df_round['Time'].apply(lambda x : date_object + timedelta(0,x))
    if start_time is not None :
        df_round = df_round[((df_round['Datetime'] > start_time) & (df_round['Datetime'] <= end_time))]
    df_round = df_round.groupby('Time').mean().reset_index()
    return df_round
  

df_round = get_df_round(None,None)
fft_plot = fft_generator([first_file],image_folder,df_round,-180,-60,-200,-10,'magma',200)



if fft_plot is None :
    fft_plot = go.Figure()

# get min date and max date for demo
date_format = "%Y-%m-%dT%H-%M-%S"
min_date ='-'.join(csv_files[0].split("data-")[-1].split('-')[:-1])
max_date ='-'.join(csv_files[-1].split("data-")[-1].split('-')[:-1])
min_date = datetime.strptime(min_date, date_format)
max_date = datetime.strptime(max_date, date_format)

time_frame = '-'.join(csv_files[0].split("/")[-1].split('-')[:-1])
fig = px.line(df_round, x='Datetime', y='Channel 1',title=f'Channel 1 over time, {time_frame}')
fig_1 = px.line(df_round, x='Datetime', y='Channel 2',title='Channel 2 over time')

fig_1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

fig_fft = fft_plot
fft_plot.update_layout(title=f"Spectrogram")


hours = [f"{i:0{2}}" for i in range(24)]
minutes = [f"{i:0{2}}" for i in range(60)]
seconds = [f"{i:0{2}}" for i in range(60)]

picker_style = {
    "display": "inline-block",
    "width": "50px",
    "cursor": "pointer",
    "border": "none",
}
separator = html.Span(":")

# Dash App Code
app = Dash(
    external_stylesheets=[dbc.themes.FLATLY]
)

app.layout = dbc.Tabs([
    dbc.Tab(
        dcc.Loading(id="loading",children=[html.Div([
        
        dbc.Row([html.H1(children='Antennas Data', style={'textAlign':'center'}),]),
        dbc.Row([html.Hr()]),

        html.H2(children='Spectrogram', style={'textAlign':'center'}),

        dbc.Row(children=[dcc.Graph(id='graph-fft',figure=fig_fft)]),
        html.H2(children='Frequency Channels', style={'textAlign':'center'}),

        dbc.Row([dcc.Graph(id='graph-channel-1',figure=fig)]),
        dbc.Row([dcc.Graph(id='graph-channel-2',figure=fig_1)]),


        ])]
        ,type="circle",
        style={
            "position": "fixed",
            "top": "200px",
            "left": "50%",
            "zIndex": "9999",
        }), 
    label="Main"),

    dbc.Tab(html.Div([ 
        
        dbc.Row([html.H2(children='Choose Channel Scale', style={'textAlign':'center'})]),
        dbc.Row([dbc.Col([html.H4(children='Channel 1')],width={"size": 5,'offset':1}),
                 
                 
                 ]),
                   

        dbc.Row([dbc.Col([ 
                        dcc.RangeSlider(-350, 0, 5,
                        value=[-180, -60],
                        id='channel-1-slider')],width={"size": 12}),]),
        dbc.Row([dbc.Col([html.H4(children='Channel 2')],width={"size": 4,'offset':1}),]),

        dbc.Row([          dbc.Col([ 
                        dcc.RangeSlider(-350, 0, 5,
                        value=[-200, -10],
                        id='channel-2-slider')],width={"size": 12}),

                 
                 
                 ]),
        dbc.Row([dbc.Col([html.H4(children='Color Scale')],width={"size": 4,'offset':1}),]),
         dbc.Row([          
             dbc.Col([

                 dcc.Dropdown(
                    id='color_dropdown', 
                    options=colorscales,
                    value=default_color_scale
                    ),
             ],width={"size": 5,'offset':1})]),
        dbc.Row([dbc.Col([html.H4(children='Custom Color Scale')],width={"size": 4,'offset':1}),]),
        dbc.Row([dbc.Col([html.P(children='E.g. black,black,red,blue,green')],width={"size": 4,'offset':1}),]),


         dbc.Row([          
             dbc.Col([

                 dcc.Textarea(
            id='textarea',
            value='',
            style={'width': '100%', 'height': 100},
                )
             ],width={"size": 5,'offset':1})]),

        dbc.Row([dbc.Col([html.H4(children='Spectrogram N per segment')],width={"size": 3,'offset':1}),]),
        dbc.Row([dbc.Col([dcc.Input(id="nperseg_input", type="number", placeholder=200, min=2,max=1000,style={'marginRight':'10px'})],
                         width={"size": 1,'offset':1})
                 ]),
        dbc.Row([html.H2(children='Choose Timeframe', style={'textAlign':'center'})]),
        dbc.Row([dbc.Col([
            dcc.DatePickerSingle(
            id='my-date-picker-single',
            min_date_allowed=min_date,
            max_date_allowed=max_date,
            initial_visible_month=min_date,
            date=min_date
            )],width={"size": 4, "offset": 3})
        ],justify="center"),
        dbc.Row([dbc.Col([html.H4(children='Start')],width={"size": 5,"offset":2}),
                 dbc.Col([html.H4(children='End')],width={"size": 4})]),
        dbc.Row([html.Hr()]),
        dbc.Row([
            dbc.Col([dcc.Dropdown(
                    hours,
                    placeholder="HH",
                    style=picker_style,id='start_hour'
                ),
                separator,
            dcc.Dropdown(minutes, placeholder="MM",         style=picker_style,id='start_minutes'),
                        separator,
            dcc.Dropdown(seconds, placeholder="SS", style=picker_style,id='start_seconds'),
                ],width={"size": 5,"offset":2}),

            dbc.Col([dcc.Dropdown(
                            hours,
                            placeholder="HH",
                            style=picker_style,
                        id='end_hour'),
                        separator,
            dcc.Dropdown(minutes, placeholder="MM",         style=picker_style,id='end_minutes'),
                        separator,
            dcc.Dropdown(seconds, placeholder="SS", style=picker_style,id='end_seconds')],width={"size": 4})
            ,
            dbc.Col(dbc.Button("Go",outline=True, color="dark", className="me-1",id='go_button'), width=1)]),
        dbc.Row([html.Hr()]),


                
        ]),
        
     label="Settings")
])



@app.callback(
    [Output('graph-channel-1', 'figure'),
     Output('graph-channel-2', 'figure'),
     Output('graph-fft', 'figure'),
     ], 
     [
 
      Input("go_button", "n_clicks"),
      Input('start_hour', "value"),
      Input("start_minutes", "value"),
      Input("start_seconds", "value"),
      Input("end_hour", "value"),
      Input("end_minutes", "value"),
      Input("end_seconds", "value"),
      Input("my-date-picker-single", "date"),
      Input('channel-1-slider', 'value'),
      Input('channel-2-slider', 'value'),
      Input('color_dropdown', 'value'),
      Input('textarea', 'value'),
      Input('nperseg_input', 'value'),
      ]
)
def on_button_click(go,start_hour,start_min,start_sec,end_hour,end_minutes,end_seconds,date_selected,
                    channel_1_slider,channel_2_slider,scale,custom_scale_text,nperseg):
    """
    Updates the spectrogram graphs as well as the frequency graphs based on the settings page.

    """
    changed_id = [p['prop_id'] for p in ctx.triggered][0]

    if 'go_button.n_clicks' in changed_id:
        if start_hour is None : 
            start_time = None
            end_time = None
        else :
            date_selected = datetime.strptime(date_selected,"%Y-%m-%dT%H:%M:%S")
            start_time = datetime.combine(date_selected,time(int(start_hour), int(start_min), int(start_sec)))
            end_time = datetime.combine(date_selected,time(int(end_hour), int(end_minutes), int(end_seconds)))
    else :
        start_time = None
        end_time = None

    csv_files = get_files(start_time,end_time)

    if start_time is None :
        time_frame = '-'.join(csv_files[0].split("/")[-1].split('-')[:-1])
    else :
        time_frame = str(start_time) +'to' + str(end_time)
           

    if start_time is not None :

        df_round = get_df_round(start_time,end_time)
    else :
        df_round = get_df_round(None,None)    

    fig_new = px.line(df_round, x='Datetime', y='Channel 1',title=f'Channel 1 over time, {time_frame}')
    fig_new_1 = px.line(df_round, x='Datetime', y='Channel 2',title=f'Channel 2 over time')

    if custom_scale_text != '' : 
        scale = custom_scale_text.split(',')
    
    fft_plot = fft_generator(csv_files,image_folder,df_round,channel_1_slider[0],
                             channel_1_slider[1],channel_2_slider[0],channel_2_slider[1],
                             scale,nperseg)
    fft_plot.update_layout(title="")

    return fig_new,fig_new_1,fft_plot


    

    

if __name__ == '__main__':
    app.run(debug=True)