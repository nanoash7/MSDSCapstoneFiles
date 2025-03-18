import os

def count_csv_files(directory):
    count = 0
    for root, _, files in os.walk(directory):
        count += sum(1 for file in files if file.endswith('.parquet'))
    return count

# Example usage
directory_path = "E:\TDVS"
csv_count = count_csv_files(directory_path)
print(f"Number of CSV files: {csv_count}")