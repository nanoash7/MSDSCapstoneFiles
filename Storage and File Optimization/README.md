# Storage and File Optimization Code

This section of code pertains to file storage and file reading optimizations. 13TB of CSV files is both high in storage space and very slow to read. Converting this data into
the parquet file format yields a 5x reduction in storage space and a 6x reduction in reading time.

csv_to_parquet.py: This file contains the code to convert a nested directory of csv files into parquet files. 
The converted files are created in a new user specified directory that contains the same nested file structure as the original directory.

file_counter.py: This file contains a small snippet of code to count the number of files in a given directory. 

multi_signal_processing_parquet.py: This is the file that contains the post-processing code. The original file (that was provided to us by the lab) was written with
csv files as the desired data source. I modified the code in this file to handle parquet files.
