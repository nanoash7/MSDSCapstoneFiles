import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def convert_file(csv_path: str, parquet_path: str):
    """
    Convert a single CSV file to Parquet.
    
    :param csv_path: Path to the CSV file.
    :param parquet_path: Path to save the Parquet file.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)  # Avoid dtype inference slowdown
        df.to_parquet(parquet_path, engine="pyarrow", index=False)  # Faster Parquet writing
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

def process_directory(source_root: str, target_root: str):
    """
    Convert all CSV files in the source_root directory to Parquet format in parallel.
    
    :param source_root: Root directory containing CSV files.
    :param target_root: Root directory for converted Parquet files.
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    
    tasks = []
    with ProcessPoolExecutor() as executor:
        for dirpath, _, filenames in os.walk(source_root):
            relative_path = Path(dirpath).relative_to(source_root)
            target_dir = target_root / relative_path
            target_dir.mkdir(parents=True, exist_ok=True)

            for filename in filenames:
                if filename.endswith(".csv"):
                    csv_path = Path(dirpath) / filename
                    parquet_path = target_dir / (filename.rsplit(".", 1)[0] + ".parquet")
                    tasks.append(executor.submit(convert_file, csv_path, parquet_path))

        # Wait for all tasks to complete
        for task in tasks:
            task.result()

if __name__ == "__main__":
    source_directory = "example_data"
    target_directory = "example_data_parquet"
    process_directory(source_directory, target_directory)
