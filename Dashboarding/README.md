# Antennas Dashboard

Antennas Dashboard is a plotly dashboard that can be run to visualize multiple Antennas frequency plots at once!

## Files

main.py - Main code to run the dashboard in your browser 
data_processing/B_FFT_Graph_Generator/FFT_Generator_dashboard.py - UW Seal code edited to create plotly spectrograms instead of matplotlib based figures
config.json - config file to set image and dataset folders for the dashboard to read from
antennas_dashboard.yml - Conda environment file to download packages for the dashboard.



## Installation

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to install Antennas Dashboard.

```bash
conda env create -f antennas_dashboard.yml
```
Activate the environment
```bash
conda activate antennas_dashboard
```


## Usage
To run the dashboard just run the command below and open the link in your browser.

```bash
python3 main.py
```

