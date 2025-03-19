
#import dask.dataframe as dd
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


matplotlib.use('Agg')
import gc
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram


def load_data(csv_file):

    data = pd.read_csv(csv_file)

    return data

# Perform FFT and create a spectrogram for a given channel
def create_spectrogram(time, voltage, min_freq, max_freq,nperseg_plot):
    # Assuming equally spaced time intervals
    fs = round(1 / (time[1] - time[0]))  # Calculate the sample rate, Hz
    if fs == 0 :
        
        fs = 100000

    
    
    
    noverlap = None
    window = None
    
     
    f, t, Sxx = spectrogram(voltage,fs=fs,nperseg=nperseg_plot)
    
    min_freq_limit = np.where(f >= min_freq)[0][0]
    freq_limit = np.where(f <= max_freq)[0][-1]
   
    
    
    #return f[min_freq_limit:freq_limit+1], t, Sxx[min_freq_limit:freq_limit+1, :]#, delta_f, delta_t
    return f, t, Sxx

def plot_chunk(ax, t, f, Sxx, start_idx, end_idx, cmap, vmin, vmax, title,df,x_color,chunk_size,color_scale):
    t_chunk = t[start_idx:end_idx]
    Sxx_chunk = Sxx[:, start_idx:end_idx]
    skip_time = len(df['Datetime']) // chunk_size
    df_min_time = float(df['Time'].min())
    t_chunk = t_chunk + df_min_time


    
    date_vals = df.copy()
    date_vals = date_vals[date_vals['Time'].isin(t_chunk)]
    
    fig_plotly = go.Heatmap(x=date_vals['Datetime'].values, y=f, z=10 * np.log10(Sxx_chunk), colorscale=color_scale,
        colorbar_x=x_color,zmin=vmin,zmax=vmax, xperiodalignment='end')#go.Figure(data=[go.Heatmap(x=t_chunk, y=f, z=10 * np.log10(Sxx_chunk), colorscale='Viridis')])
    return fig_plotly
    
def plot_spectrogram(f1, t1, Sxx1, title1, f2, t2, Sxx2, title2, \
    vmin_channel_1, vmax_channel_1, vmin_channel_2, vmax_channel_2, 
    image_name_start, image_name_end, min_freq, max_freq, image_folder,df,color_scale):
    
    colors = ["black", "black", "blue", "blue", "green", "yellow", "orange"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N = 100)
    
    # Initialize the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Define chunk size and calculate number of chunks
    chunk_size = 100  # Adjust this based on your data
    num_chunks = int(np.ceil(len(t1) / chunk_size))

    # Plot each chunk
    for i in range(1):
        
        start_idx = i * chunk_size
        end_idx = len(t1)
    

        pcm1 = plot_chunk(ax1, t1, f1, Sxx1, start_idx, end_idx, cmap, vmin_channel_1, vmax_channel_1, title1,df,1,chunk_size,color_scale)
        pcm2 = plot_chunk(ax2, t2, f2, Sxx2, start_idx, end_idx, cmap, vmin_channel_2, vmax_channel_2, title2,df,1.1,chunk_size,color_scale)
        return pcm1,pcm2
    
    fig = make_subplots(rows=2, cols=1,subplot_titles=(title1,title2))

    fig.add_trace(
        pcm1,
        row=1, col=1,
    )

    fig.add_trace(
        pcm2,
        row=2, col=1
    )

    fig.update_layout(height=1000, width=1800, title_text=f'TestResult- {image_name_start} - {image_name_end} - {min_freq} to {max_freq}')



    return fig

# Main function to load data and create spectrograms for both channels
def plot_main(time, voltage1, voltage2, min_freq, max_freq, \
    vmin_channel_1, vmax_channel_1, vmin_channel_2, vmax_channel_2, image_name_start, 
    image_name_end, image_folder,df,color_scale,nperseg_plot):
    # Create and plot spectrogram for the first channel
    f1, t1, Sxx1 = create_spectrogram(time, voltage1, min_freq, max_freq,nperseg_plot)
    

    
    # Create and plot spectrogram for the second channel
    f2, t2, Sxx2 = create_spectrogram(time, voltage2, min_freq, max_freq,nperseg_plot)
    

    
    fig1,fig2 = plot_spectrogram(f1, t1, Sxx1, f'Channel 1 Spectrogram | Scale: min: {vmin_channel_1}, max: {vmax_channel_1}', f2, t2, Sxx2, \
        f'Channel 2 Spectrogram | Scale: min: {vmin_channel_2} max: {vmax_channel_2}', vmin_channel_1, vmax_channel_1, vmin_channel_2, vmax_channel_2, \
            image_name_start, image_name_end, min_freq, max_freq, image_folder,df,color_scale)
    return fig1,fig2

def print_progress_bar(iteration, total, prefix='', length=None, fill='#'):
    if length is None:
        # Get the current terminal width
        columns = shutil.get_terminal_size().columns
        # Adjust the length of the bar to fit the terminal width
        length = max(columns - len(prefix) - 50, 70)  # adjust 50 based on other text length

    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r{} |{}| {}%'.format(prefix, bar, percent))


def main(files, image_folder,df,vmin_channel_1,vmax_channel_1,
         vmin_channel_2,vmax_channel_2,color_scale,nperseg_plot):
    max_freq = 200 # Set the maximum frequency
    min_freq = 0  # Set the minimum frequency

    image_folder = image_folder#"C:/Users/Embedded Group/Desktop/image_extract/downsample and notch filter"
    # Define the number of files to process at a time
    n = len(files) # Adjust this number as needed
    if nperseg_plot is None : 
        nperseg_plot = 200
    if vmax_channel_1 is None : 

        vmin_channel_1 = -180
        vmax_channel_1 = -60
        vmin_channel_2 = -200
        vmax_channel_2 = -10
    fig = None

    time = np.array([])
    time = df['Time'].values
    if len(time) == 0 :
        return fig
        
    date_format = "%Y-%m-%dT%H-%M-%S"

    image_name_start = datetime.strptime(files[0].split('/')[-1][5:-11], date_format)
    image_name_end = datetime.strptime(files[-1].split('/')[-1][5:-11], date_format)
    
    voltage_df = pd.DataFrame()

    for file_name in files:
        if file_name.endswith('.csv') :
            voltage_df = pd.concat([voltage_df,pd.read_csv(file_name)])
            
    voltage1 = df['Channel 1'].values
    voltage2 = df['Channel 2'].values
    
    fig1,fig2 = plot_main(time, voltage1, voltage2, min_freq, max_freq, \
        vmin_channel_1, vmax_channel_1, vmin_channel_2, vmax_channel_2, 
        image_name_start, image_name_end, image_folder,df,color_scale,nperseg_plot)
    return fig1,fig2

if __name__ == '__main__' :
    main()