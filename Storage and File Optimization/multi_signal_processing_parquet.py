import json
import math
import io
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction
from multiprocessing import Pool, Manager, cpu_count
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

try :
    from .FFT_Generator import create_spectrogram
except: 
    from FFT_Generator import create_spectrogram


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# load input parameters
def load_input_parameters_json() :
    # record current dir
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to data.json
    data_json_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data_plot_input.json')

    with open(data_json_path, 'r') as file:
        data = json.load(file)
    return data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandstop', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, pad_width=None):
    # If pad_width is None or not provided, default to 10% of data length or minimum 50 points
    if pad_width is None:
        pad_width = max(len(data) // 10, 50)
    
    # Mirror padding
    pad_data = np.pad(data, pad_width, mode='reflect')
    
    # Apply the filter to the padded data
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_padded_data = sosfiltfilt(sos, pad_data, padlen=pad_width)
    
    # Remove padding after filtering
    y = filtered_padded_data[pad_width:-pad_width]
    
    return y

def downsample(data, sampling_rate, frequency_filter = 0, fs = 5000, order = 6, frequency_flag = True):
    time = data.iloc[:, 0].values
    channel_1 = data.iloc[:, 1].values
    channel_2 = data.iloc[:, 2].values
    
    ratio = Fraction(fs, sampling_rate)
        
    # The numerator is the upsampling factor, the denominator is the downsampling factor
    up_factor = ratio.numerator
    down_factor = ratio.denominator

    channel_1 = multi_downsampling(channel_1, sampling_rate // fs, up_factor, down_factor)
    channel_2 = multi_downsampling(channel_2, sampling_rate // fs, up_factor, down_factor)
    
    if frequency_filter != 0 :

        lowcut = frequency_filter - 5  # Lower bound of the desired frequency range
        highcut = frequency_filter + 5  # Upper bound of the desired frequency range

    if frequency_flag :
        filtered_signal_1 = np.concatenate((channel_1[:1], butter_bandpass_filter(channel_1, lowcut, highcut, fs, order)[1:]))
        filtered_signal_2 = np.concatenate((channel_2[:1], butter_bandpass_filter(channel_2, lowcut, highcut, fs, order)[1:]))

        # time_downsampled = time[::(sampling_rate // fs)]

        # print(len(time_downsampled))
        print(len(filtered_signal_1))
        print(len(filtered_signal_2))

        df = pd.DataFrame({
            # 'Time': time_downsampled,
            'Channel 0': filtered_signal_1,
            'Channel 1': filtered_signal_2
        })
    else :
        # time_downsampled = time[::(sampling_rate // fs)]

        df = pd.DataFrame({
            # 'Time': time_downsampled,
            'Channel 0': channel_1,
            'Channel 1': channel_2
        })

    return df

def multi_downsampling(data, factor, up1, down1):
    # return decimate(data, factor, ftype='fir', zero_phase=True)
    return resample_poly(data, up1, down1)

def plot(ax, t, f, Sxx, cmap, vmin, vmax, title, rate) :
    pcm = ax.pcolormesh(t, f, 20 * np.log10(Sxx), cmap = cmap, vmin = vmin, vmax = vmax)
    # pcm = ax.pcolormesh(t, f, 20 * np.log10(Sxx), vmin = vmin, vmax = vmax)
    # pcm = ax.specgram(10 * np.log10(Sxx), Fs=rate)
    ax.set_title(title, fontsize = 10)
    return pcm

def plot_adjusted(ax, t, f, Sxx, cmap, vmin, vmax, title, rate) :
    pcm = ax.pcolormesh(t, f, Sxx, cmap = cmap, vmin = vmin, vmax = vmax)
    # pcm = ax.specgram(10 * np.log10(Sxx), Fs=rate)
    ax.set_title(title, fontsize = 14)
    return pcm

        
def plot_method(delta_f, delta_t, min1_intensity, max1_intensity, min2_intensity, max2_intensity,
                file_name, chan1_data, chan2_data, sampling_rate, min_freq, max_freq, cmap, image_folder, excel) :
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    if excel :
        f1, t1, Sxx1, excel_data_1 = chan1_data
        f2, t2, Sxx2, excel_data_2 = chan2_data
    else :
        f1, t1, Sxx1 = chan1_data
        f2, t2, Sxx2 = chan2_data
    
    # num_segments = 120
    # points_per_segment = len(t1) // num_segments

    # for i in range(num_segments):
    #     start_idx = i * points_per_segment
    #     end_idx = start_idx + points_per_segment if i < num_segments - 1 else len(t1)

    #     # Extracting the segment for both channels
    #     t1_segment = t1[start_idx:end_idx]
    #     Sxx1_segment = Sxx1[:, start_idx:end_idx]
    #     t2_segment = t2[start_idx:end_idx]
    #     Sxx2_segment = Sxx2[:, start_idx:end_idx]

    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    #     # Plot each segment
    #     pcm1 = plot(ax1, t1_segment, f1, Sxx1_segment, cmap, min1_intensity, max1_intensity, f'Channel 0 Spectrogram | Scale: min: {int(min1_intensity)}, max: {int(max1_intensity)}', sampling_rate)
    #     pcm2 = plot(ax2, t2_segment, f2, Sxx2_segment, cmap, min2_intensity, max2_intensity, f'Channel 1 Spectrogram | Scale: min: {int(min2_intensity)} max: {int(max2_intensity)}', sampling_rate)

    #     # Colorbars
    #     cbar1 = plt.colorbar(pcm1, ax=ax1)
    #     cbar1.set_label('Intensity [dB]')
    #     cbar2 = plt.colorbar(pcm2, ax=ax2)
    #     cbar2.set_label('Intensity [dB]')

    #     # Save each figure
    #     image_file = f'Segment_{i+1}_{file_name[5:-4]}.png'
    #     plt.savefig(os.path.join(image_folder, image_file))
    #     plt.close(fig)

    #     print(f'Segment {i+1} image saved successfully!')
    
    pcm1 = plot(ax1, t1, f1, Sxx1, cmap, min1_intensity, max1_intensity, f'Channel 0 Spectrogram | Scale: min: {int(min1_intensity)}, max: {int(max1_intensity)} | Δf = {delta_f} Hz, Δt = {delta_t} s', sampling_rate)
    pcm2 = plot(ax2, t2, f2, Sxx2, cmap, min2_intensity, max2_intensity, f'Channel 1 Spectrogram | Scale: min: {int(min2_intensity)} max: {int(max2_intensity)} | Δf = {delta_f} Hz, Δt = {delta_t} s', sampling_rate)

    # Set labels and colorbars for both subplots
    ax1.set_ylabel('Frequency [Hz]', fontsize = 12)
    ax1.set_xlabel('Time [sec]', fontsize = 12)
    ax2.set_ylabel('Frequency [Hz]', fontsize = 12)
    ax2.set_xlabel('Time [sec]', fontsize = 12)
    ax1.tick_params(labelsize = 10)
    ax2.tick_params(labelsize = 10)

    # Add colorbars
    cbar1 = plt.colorbar(pcm1, ax=ax1)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.set_label('Intensity [dB]', fontsize = 12)
    cbar2 = plt.colorbar(pcm2, ax=ax2)
    cbar2.ax.tick_params(labelsize=10)
    cbar2.set_label('Intensity [dB]', fontsize=12)

    plt.tight_layout(pad=3.0)
    # file = file.replace('.csv', '')
    image_file = f'TestResult-start-{file_name[5:]}-{min_freq}-to-{max_freq}.png'
    plt.savefig(os.path.join(image_folder, image_file))
    
    print('image saved successfully!')
    plt.close(fig)
    
    if excel :
        # df = pd.DataFrame({
        #     'TimeStamp': t1,
        #     'channel 0': chan1_column_mean,
        #     'channel 1': chan2_column_mean
        # })
        
        excel_file_1 = f"excel-{file_name[5:]}-{min_freq}-to-{max_freq}-1.csv"
        excel_file_2 = f"excel-{file_name[5:]}-{min_freq}-to-{max_freq}-2.csv"
        
        combined_1 = np.column_stack((f1, excel_data_1))
        combined_2 = np.column_stack((f2, excel_data_2))
        
        # combined_1 = np.column_stack((f1, Sxx1))
        # combined_2 = np.column_stack((f2, Sxx2))
        
        np.savetxt(os.path.join(os.path.join(image_folder, "excels"), excel_file_1), combined_1, delimiter=",")
        np.savetxt(os.path.join(os.path.join(image_folder, "excels"), excel_file_2), combined_2, delimiter=",")
        
        # excel_data_1.to_csv(os.path.join(os.path.join(image_folder, "excels"), excel_file_1))
        # excel_data_2.to_csv(os.path.join(os.path.join(image_folder, "excels"), excel_file_2))
    
    
def raw_data_reading(args):
    file_batch, file_folder, rate_to_10 = args
    file_name = file_batch[0]
    df_list = []
    
    for file in file_batch:
        if rate_to_10 > 1:
            # Read Parquet file instead of CSV
            df = pd.read_parquet(os.path.join(file_folder, file))
            total_rows = len(df)
    
            part_size = total_rows // rate_to_10
            
            for i in range(rate_to_10):
                start = i * part_size
                # If it's the last part, include all remaining rows
                if i == rate_to_10 - 1:
                    end = total_rows
                else:
                    end = start + part_size
                
                # Slice the DataFrame and append the part
                df_list.append(df.iloc[start:end])
        else:
            # Read the entire Parquet file if rate_to_10 is 1
            df = pd.read_parquet(os.path.join(file_folder, file))
            df_list.append(df)
        
        print(f'Reading file {file}')
    
    return {file_name: df_list}
        
        
def raw_data_process(df, downsample_flag, sampling_rate, frequency_filtered,
                    down_to_sampling_rate, frequency_flag,
                    min_freq, max_freq, rate_to_10, excel) :
    
    if rate_to_10 > 1 :
        filename = ""
        idx = 0
        returned_list = []
        for file_name in df :
            filename = file_name
            for df_item in df[file_name] :
                if downsample_flag :
                    if frequency_flag :    
                        df_downsample = downsample(df_item, sampling_rate, frequency_filtered, down_to_sampling_rate, 6, frequency_flag)
                    else :
                        df_downsample = downsample(df_item, sampling_rate, 0, down_to_sampling_rate, 6, frequency_flag)        
                    # time = df_downsample.iloc[:, 0].values
                    channel_1 = df_downsample.iloc[:, 0].values
                    channel_2 = df_downsample.iloc[:, 1].values
                else :
                    # time = df_item.iloc[:, 0].values
                    channel_1 = df_item.iloc[:, 0].values
                    channel_2 = df_item.iloc[:, 1].values

                f1, t1, Sxx1, delta_f, delta_t = create_spectrogram(None, channel_1, min_freq, max_freq, down_to_sampling_rate)

                f2, t2, Sxx2, delta_f, delta_t = create_spectrogram(None, channel_2, min_freq, max_freq, down_to_sampling_rate)
                tmp_file_name = file_name.replace(".csv", "")
                filename = f"{tmp_file_name}-{idx}"
                idx += 1
                
                if excel :
                # returned_list.append((delta_f, delta_t, {filename: (f1, t1, Sxx1)}, {filename: (f2, t2, Sxx2)}, np.percentile(20 * np.log10(Sxx1), 10), np.max(20 * np.log10(Sxx1)), np.percentile(20 * np.log10(Sxx2), 10), np.max(20 * np.log10(Sxx2)), filename))
                    # returned_list.append((delta_f, delta_t, {filename: (f1, t1, Sxx1, np.mean(20 * np.log10(Sxx1), axis = 1))}, {filename: (f2, t2, Sxx2, np.mean(20 * np.log10(Sxx2), axis = 1))}, np.min(20 * np.log10(Sxx1)), np.max(20 * np.log10(Sxx1)), np.min(20 * np.log10(Sxx2)), np.max(20 * np.log10(Sxx2)), filename))
                    returned_list.append((delta_f, delta_t, {filename: (f1, t1, Sxx1, 20 * np.log10(Sxx1))}, {filename: (f2, t2, Sxx2, 20 * np.log10(Sxx2))}, np.min(20 * np.log10(Sxx1)), np.max(20 * np.log10(Sxx1)), np.min(20 * np.log10(Sxx2)), np.max(20 * np.log10(Sxx2)), filename))
                else :
                    returned_list.append((delta_f, delta_t, {filename: (f1, t1, Sxx1)}, {filename: (f2, t2, Sxx2)}, np.percentile(20 * np.log10(Sxx1), 10), np.max(20 * np.log10(Sxx1)), np.percentile(20 * np.log10(Sxx2), 10), np.max(20 * np.log10(Sxx2)), filename))

        return returned_list
    else :
        spec_1 = pd.DataFrame()
        filename = ""
        for file_name in df :
            filename = file_name
            for df_item in df[file_name] :
                spec_1 = pd.concat([spec_1, df_item], ignore_index = True)
        filename = filename.replace(".csv", "")
        
        if downsample_flag :
            if frequency_flag :    
                df_downsample = downsample(spec_1, sampling_rate, frequency_filtered, down_to_sampling_rate, 6, frequency_flag)
            else :
                df_downsample = downsample(spec_1, sampling_rate, 0, down_to_sampling_rate, 6, frequency_flag)
            # time = df_downsample.iloc[:, 0].values
            channel_1 = df_downsample.iloc[:, 0].values
            channel_2 = df_downsample.iloc[:, 1].values
        else :
            # time = spec_1.iloc[:, 0].values
            channel_1 = spec_1.iloc[:, 0].values
            channel_2 = spec_1.iloc[:, 1].values

        f1, t1, Sxx1, delta_f, delta_t = create_spectrogram(None, channel_1, min_freq, max_freq, down_to_sampling_rate)

        f2, t2, Sxx2, delta_f, delta_t = create_spectrogram(None, channel_2, min_freq, max_freq, down_to_sampling_rate)
        
        if excel :
            return (delta_f, delta_t, {filename: (f1, t1, Sxx1, 20 * np.log10(Sxx1))}, {filename: (f2, t2, Sxx2, 20 * np.log10(Sxx2))}, np.min(20 * np.log10(Sxx1)), np.max(20 * np.log10(Sxx1)), np.min(20 * np.log10(Sxx2)), np.max(20 * np.log10(Sxx2)), filename)
            # return (delta_f, delta_t, {filename: (f1, t1, Sxx1, np.mean(20 * np.log10(Sxx1), axis = 1))}, {filename: (f2, t2, Sxx2, np.mean(20 * np.log10(Sxx2), axis = 1))}, np.min(20 * np.log10(Sxx1)), np.max(20 * np.log10(Sxx1)), np.min(20 * np.log10(Sxx2)), np.max(20 * np.log10(Sxx2)), filename)
        else  :
            return (delta_f, delta_t, {filename: (f1, t1, Sxx1)}, {filename: (f2, t2, Sxx2)}, np.min(20 * np.log10(Sxx1)), np.max(20 * np.log10(Sxx1)), np.min(20 * np.log10(Sxx2)), np.max(20 * np.log10(Sxx2)), filename)
        # return (delta_f, delta_t, {filename: (f1, t1, Sxx1)}, {filename: (f2, t2, Sxx2)}, np.percentile(20 * np.log10(Sxx1), 10), np.max(20 * np.log10(Sxx1)), np.percentile(20 * np.log10(Sxx2), 10), np.max(20 * np.log10(Sxx2)), filename)

def chunk_files(files, chunk_size):
    """Yield successive chunk_size chunks from files."""
    for i in range(0, len(files), chunk_size):
        yield files[i:i + chunk_size]
        
def extract_timestamp(filename):
    # example format: data-2024-07-27T01-43-59-879042.csv
    timestamp_str = filename.split('-')[1:6]
    print(timestamp_str)
    timestamp_str = '-'.join(timestamp_str)
    timestamp_str = timestamp_str.replace('-', ':')  # Adjust to proper datetime format
    return datetime.strptime(timestamp_str, "%Y:%m:%dT%H:%M:%S")

def calculate_index(start_filename, target_filename, interval_seconds=10):
    # Extract datetime objects from filenames
    start_time = extract_timestamp(start_filename)
    target_time = extract_timestamp(target_filename)
    
    print(start_time)
    print(target_time)
    
    time_difference = target_time - start_time
    
    total_seconds = time_difference.total_seconds()
    
    index_offset = int(total_seconds / interval_seconds) + 1
    
    return index_offset

if __name__ == '__main__' :
    parameters = load_input_parameters_json()
    sampling_rate = int(parameters["sampling rate (only supports 100k Hz now)"])
    lines_per_data_file = 1000000
    
    minutes_before_and_after = int(parameters["minutes before and after"])
    seconds_before_and_after = int(parameters["seconds before and after"])
    
    if minutes_before_and_after == 0 and seconds_before_and_after == 0 :
        raise Exception("unspecified plotting time range")
    
    if minutes_before_and_after != 0 and seconds_before_and_after != 0 :
        raise Exception("overspecified plotting time range, please set either minutes before and after or seconds before and after to 0")

    if minutes_before_and_after :
        file_nums = math.ceil(minutes_before_and_after * 60 / (lines_per_data_file // sampling_rate))
    else :
        file_nums = math.ceil(seconds_before_and_after / (lines_per_data_file // sampling_rate))
    file_list = [] # list of files that include the time range

    meteor_event_time = parameters["meteor event time"]
    
    # center_files = parameters["center files"]
    data_folders = parameters["data-folders"]
    
    # center_file_list = center_files.split("\n")
    data_folder_list = data_folders.split("\n")
    # if len(center_file_list) != len(data_folder_list) :
    #     raise Exception("number of center files does not math number of data folders")
    # task_dict = {}
    # for idx, center_file in enumerate(center_file_list, 0) :
    #     task_dict[center_file] = data_folder_list[idx]
        
    file_folder = data_folder_list[0]
    data_file_list = os.listdir(file_folder)
    data_file_list.sort(key=natural_keys)
    millisecond_stamp = data_file_list[0].split('-')[-1:]
    meteor_file_name = f'data-2024-{meteor_event_time}-{millisecond_stamp}'
    meteor_index = calculate_index(data_file_list[0], meteor_file_name)

    print(meteor_index - file_nums)
    print(meteor_index + file_nums + 1)
    print(len(data_file_list))
    plot_data_file_list = data_file_list[max(0, meteor_index - file_nums - 1) : min(meteor_index + file_nums + 1, len(data_file_list))]
    
    # print(plot_data_file_list)
    

    storage_dir = parameters["image storage dir"]
    image_folder = os.path.join(storage_dir, meteor_event_time)
    os.makedirs(image_folder, exist_ok = True)

    # colors = ["black", "darkgreen", "green", "yellow", "yellow", "orange", "red", "darkred"]
    # v =      [   0   ,     0.1    ,  .30   ,   .60   ,   .75   ,   .77   ,  .85 ,    1.0   ]
    
    colors = re.split(r'\s*,\s*', parameters["colors"])
    v_str_list = re.split(r'\s*,\s*', parameters["colors pos"])
    v = [float(v_item) for v_item in v_str_list]
    l = list(zip(v, colors))
    cmap = LinearSegmentedColormap.from_list("custom_cmap", l)
    max_freq = int(parameters["upper range frequency (Hz)"]) # Set the maximum frequency
    min_freq = int(parameters["lower range frequency (Hz)"])  # Set the minimum frequency
    
    if int(parameters["final sampling rate (if downsample, Hz)"]) == 0 and int(parameters["frequncy to be filtered (Hz)"]) != 0 :
        raise Exception("Warning: you are trying to filter out signal before downsampling, which is not a proper thing to do!! (if you are only filtering out low frequency)")
    
    if int(parameters["frequncy to be filtered (Hz)"]) != 0 :
        frequency_filtered = int(parameters["frequncy to be filtered (Hz)"])
        frequency_flag = True
    else :
        frequency_filtered = 0
        frequency_flag = False
    if int(parameters["final sampling rate (if downsample, Hz)"]) != 0 :
        down_to_sampling_rate = int(parameters["final sampling rate (if downsample, Hz)"])
        downsample_flag = True
    else :
        down_to_sampling_rate = 0
        downsample_flag = False
    order = 6

    # file_folder = data_folder_list[0]
    # file_list = os.listdir(file_folder)
    # # file_folders = os.listdir(file_folder)
    # # file_folders.sort(key=natural_keys)
    # file_list.sort(key = natural_keys)
        
    rate_to_10 = 1.0
    valid_less_than_10 = [1, 2, 5]
    duration_per_img = int(parameters["duration per img (seconds)"])
    if duration_per_img <= 0 :
        raise Exception("Invalid duration per image")
    if duration_per_img >= 10 :
        num_of_files = int(duration_per_img // 10)
        file_batches = list(chunk_files(plot_data_file_list, num_of_files))
    elif duration_per_img < 10 :
        if duration_per_img in valid_less_than_10 :
            rate_to_10 = 10 / duration_per_img
            file_batches = list(chunk_files(plot_data_file_list, 1))
        else :
            raise Exception("Invalid duration per image")
    rate_to_10 = int(rate_to_10)
    #print(file_batches)
    print(duration_per_img)
    
    excel_or_not : str = parameters["excel generate"]
    excel_or_not : str = excel_or_not.lower()
    
    if excel_or_not != "y" and excel_or_not != "n" :
        raise Exception("Invalid excel or not input")
    elif excel_or_not == "y" :
        excel = True
        os.makedirs(os.path.join(image_folder, "excels"), exist_ok=True)
    else :
        excel = False
    
    # print(file_batches)
    args_list = [(file_batch, file_folder, rate_to_10) for file_batch in file_batches]
    # args_list = [(file_batch, file_folder) for file_batch in file_folders]
    
    print("start of parquet reads")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers = 4096) as executor:
        df_list = list(executor.map(raw_data_reading, args_list))
    # df_list = raw_data_reading(file_folder)
    print("end of parquet reads")
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    quit()
    
        
    process_args_list = [(df, downsample_flag, sampling_rate, frequency_filtered,
                        down_to_sampling_rate, frequency_flag,
                        min_freq, max_freq, rate_to_10, excel) for df in df_list]
        
    with Pool(processes = 60) as pool:
        # Using starmap to pass multiple arguments
        results = pool.starmap(raw_data_process, process_args_list)

    chan1_dict = {}
    chan2_dict = {}
    min1_buffer = []
    max1_buffer = []
    min2_buffer = []
    max2_buffer = []
    file_names = []
    # {filename: (f1, t1, Sxx1)}, {filename: (f2, t2, Sxx2)}, np.min(Sxx1), np.max(Sxx1), np.min(Sxx2), np.max(Sxx2), filename
    if type(results[0]) is list :
        for result_list in results :
            for result in result_list :
                delta_f, delta_t, chan1_pair, chan2_pair, min1, max1, min2, max2, file_name = result
                chan1_dict.update(chan1_pair)
                chan2_dict.update(chan2_pair)
                min1_buffer.append(min1)
                max1_buffer.append(max1)
                min2_buffer.append(min2)
                max2_buffer.append(max2)
                file_names.append(file_name)
    else :
        for result in results :
            delta_f, delta_t, chan1_pair, chan2_pair, min1, max1, min2, max2, file_name = result
            chan1_dict.update(chan1_pair)
            chan2_dict.update(chan2_pair)
            min1_buffer.append(min1)
            max1_buffer.append(max1)
            min2_buffer.append(min2)
            max2_buffer.append(max2)
            file_names.append(file_name)
    # global_chan1_min = min(min1_buffer)
    # global_chan1_max = max(max1_buffer)
    # global_chan2_min = min(min2_buffer)
    # global_chan2_max = max(max2_buffer)
    # global_chan1_min = np.percentile(min1_buffer, 80)
    # global_chan1_max = np.percentile(max1_buffer, 90)
    # global_chan2_min = np.percentile(min2_buffer, 80)
    # global_chan2_max = np.percentile(max2_buffer, 90)
    # global_chan1_min = np.percentile(min1_buffer, 50)
    # global_chan1_max = np.percentile(max1_buffer, 85)
    # global_chan2_min = np.percentile(min2_buffer, 50)
    # global_chan2_max = np.percentile(max2_buffer, 85)
    
    
    #7/27
    #global_chan1_min = -250
    #global_chan1_max = -140
    #global_chan2_min = -190
    #global_chan2_max = -90
    
    global_chan1_min = -240
    global_chan1_max = -150
    global_chan2_min = -180
    global_chan2_max = -120
    
    #7/28
    # global_chan1_min = -250
    # global_chan1_max = -130
    # global_chan2_min = -190
    # global_chan2_max = -80
    
    #7/29
    # global_chan1_min = -240
    # global_chan1_max = -140
    # global_chan2_min = -180
    # global_chan2_max = -90
    
    #7/30
    # global_chan1_min = -240
    # global_chan1_max = -130
    # global_chan2_min = -190
    # global_chan2_max = -80
    
    #7/31
    # global_chan1_min = -250
    # global_chan1_max = -145
    # global_chan2_min = -220
    # global_chan2_max = -115
    
    #8/01
    # global_chan1_max = -125
    # global_chan1_min = -230
    # global_chan2_max = -100
    # global_chan2_min = -210
    
    # #8/02
    # global_chan1_max = -145
    # global_chan1_min = -230
    # global_chan2_max = -120
    # global_chan2_min = -210
    
    # #8/03
    # global_chan1_max = -135
    # global_chan1_min = -240
    # global_chan2_max = -110
    # global_chan2_min = -220
    
    # 8/04
    # global_chan1_max = -140
    # global_chan1_min = -230
    # global_chan2_max = -205
    # global_chan2_min = -290
    
    # 8/05
    # global_chan1_max = -145
    # global_chan1_min = -230
    # global_chan2_max = -210
    # global_chan2_min = -300
    
    
    # #8/06
    # global_chan1_max = -130
    # global_chan1_min = -240
    # global_chan2_max = -200
    # global_chan2_min = -300
    
        
    #8/10
    # global_chan1_max = -130
    # global_chan1_min = -220
    # global_chan2_max = -220
    # global_chan2_min = -330
    
    # 8/11
    # global_chan1_max = -120
    # global_chan1_min = -220
    # global_chan2_max = -220
    # global_chan2_min = -320
    
    # 8/12
    # global_chan1_max = -120
    # global_chan1_min = -230
    # global_chan2_max = -215
    # global_chan2_min = -325
    
    # 8/13
    # global_chan1_max = -120
    # global_chan1_min = -220
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/14
    # global_chan1_max = -130
    # global_chan1_min = -250
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/15
    # global_chan1_max = -130
    # global_chan1_min = -250
    # global_chan2_max = -215
    # global_chan2_min = -325
    
    # horizontal line test
    # global_chan1_max = -300
    # global_chan1_min = -60
    # global_chan2_max = -300
    # global_chan2_min = -60

    
    # 8/16
    # global_chan1_max = -225
    # global_chan1_min = -335
    # global_chan2_max = -225
    # global_chan2_min = -325
    
    # 8/17
    # global_chan1_max = -215
    # global_chan1_min = -325
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/18
    # global_chan1_max = -215
    # global_chan1_min = -325
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/19
    # global_chan1_max = -215
    # global_chan1_min = -325
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/20
    # global_chan1_max = -215
    # global_chan1_min = -315
    # global_chan2_max = -220
    # global_chan2_min = -320
    
    # 8/21
    # global_chan1_max = -220
    # global_chan1_min = -305
    # global_chan2_max = -215
    # global_chan2_min = -315
    
    # 8/22
    # global_chan1_max = -220
    # global_chan1_min = -305
    # global_chan2_max = -215
    # global_chan2_min = -310
    
    # 8/23
    #global_chan1_max = -100
    #global_chan1_min = -180
    #global_chan2_max = -190
    #global_chan2_min = -270
    
    # 8/23 replace
    # global_chan1_max = -80
    # global_chan1_min = -170
    # global_chan2_max = -180
    # global_chan2_min = -270
    
    # 8/24
    # global_chan1_max = -190
    # global_chan1_min = -290
    # global_chan2_max = -190
    # global_chan2_min = -290
    
    # 8/25
    # global_chan1_max = -220
    # global_chan1_min = -320
    # global_chan2_max = -220
    # global_chan2_min = -320
    
    # 8/26
    # global_chan1_max = -220
    # global_chan1_min = -350
    # global_chan2_max = -220
    # global_chan2_min = -350
    
    # 8/27
    # global_chan1_max = -220
    # global_chan1_min = -320
    # global_chan2_max = -220
    # global_chan2_min = -320
    
        
    args_list = [(delta_f, delta_t, global_chan1_min, global_chan1_max, global_chan2_min, global_chan2_max,
        file_name, chan1_dict[file_name], chan2_dict[file_name], sampling_rate, min_freq, max_freq, cmap, image_folder, excel) for file_name in file_names]
        
    with Pool(processes = 60) as pool:
        # Using starmap to pass multiple arguments
        pool.starmap(plot_method, args_list)
        
        
    # data_plot_downsampled()

