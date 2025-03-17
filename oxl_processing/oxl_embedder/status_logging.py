import os 
import logging
import pandas as pd
import csv
from setting import g_base_path


embedding_base_path =  os.path.join(g_base_path, "embeddings") 
failed_download_folder = os.path.join(g_base_path, "failed_downloads")  
found_objects_folder = os.path.join(g_base_path, "found_objects")
failed_processing_folder = os.path.join(g_base_path, "failed_processing") 

def init_logging():
    """Initialize logging for the application.
    This function sets up logging for the application, creating necessary directories
    and configuring the logging system to write to a log file.
    """
    # Configure logging
    os.makedirs(embedding_base_path, exist_ok=True)
    os.makedirs(failed_download_folder, exist_ok=True)
    os.makedirs(found_objects_folder, exist_ok=True)
    os.makedirs(failed_processing_folder, exist_ok=True)
    log_file_path = os.path.join(g_base_path, 'objaverse_xl.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,  # Capture all levels of logs
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_object_to_csv(csv_filename:str, sha256: str, file_identifier: str):
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sha256", "fileIdentifier"])
            writer.writerow([sha256, file_identifier])
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([sha256, file_identifier])
        
def check_if_sha256_in_csv(csv_filename:str, sha256: str):
    if not os.path.exists(csv_filename):
        return False
    with open(csv_filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == sha256:
                return True
    return False

def create_logging_files(run_dir: str):
    # create logging files
    failed_downloads_file = os.path.join(run_dir, "failed_downloads.csv")
    if not os.path.exists(failed_downloads_file):
        with open(failed_downloads_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sha256", "fileIdentifier"])
    failed_processing_file = os.path.join(run_dir, "failed_processing.csv")
    if not os.path.exists(failed_processing_file):
        with open(failed_processing_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sha256", "fileIdentifier"])
    failed_rendering_file = os.path.join(run_dir, "failed_rendering.csv")
    if not os.path.exists(failed_rendering_file):
        with open(failed_rendering_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sha256", "fileIdentifier"])
    failed_embedding_generation_file = os.path.join(run_dir, "failed_embedding_generation.csv")
    if not os.path.exists(failed_embedding_generation_file):
        with open(failed_embedding_generation_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sha256", "fileIdentifier"])

def process_download_and_processing_failure(sample: pd.DataFrame, run_dir: str):
    # go through sample and safe each failed download and processing into the csv files
    failed_downloads_to_safe = []
    failed_processing_to_safe = []
    for index, row in sample.iterrows():
        sha256 = row["sha256"]
        file_identifier = row["fileIdentifier"]
        if os.path.isfile(os.path.join(failed_download_folder, f"{sha256}.csv")):
            failed_downloads_to_safe.append([sha256, file_identifier])
            os.remove(os.path.join(failed_download_folder, f"{sha256}.csv"))
        if os.path.isfile(os.path.join(failed_processing_folder, f"{sha256}.csv")):
            failed_processing_to_safe.append([sha256, file_identifier])
            os.remove(os.path.join(failed_processing_folder, f"{sha256}.csv"))
    
    failed_downloads_file = os.path.join(run_dir, "failed_downloads.csv")
    with open(failed_downloads_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for entry in failed_downloads_to_safe:
            writer.writerow(entry)
    failed_processing_file = os.path.join(run_dir, "failed_processing.csv")
    with open(failed_processing_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        for entry in failed_processing_to_safe:
            writer.writerow(entry)


class SummaryStatistics:
    """Class for summarizing object processing statistics.

    This class provides methods to calculate and store statistics about the processing
    of Objaverse-XL objects, including download, processing, rendering, and embedding generation.
    It also allows for updating and saving the statistics to a CSV file.
    """
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.n_samples = 0
        self.n_failed_downloads = 0
        self.n_failed_processing = 0
        self.n_failed_rendering = 0
        self.n_failed_embedding_generation = 0
        self.n_successful_embedding_generation = 0
        self.object_processing_df = None

    def __str__(self):
        return f"Number of objects to download: {self.n_samples} \n" + \
               f"Number of failed downloads: {self.n_failed_downloads} \n" + \
               f"Number of failed processing: {self.n_failed_processing} \n" + \
               f"Number of failed rendering: {self.n_failed_rendering} \n" + \
               f"Number of failed embedding generation: {self.n_failed_embedding_generation} \n" + \
               f"Number of successful embedding generation: {self.n_successful_embedding_generation} \n"
    
    def object_processing_df_to_csv(self, file_path: str):
        self.object_processing_df.to_csv(file_path, index=False)

    def update_download_rendering_statistics(self):
        self.n_failed_downloads, _ = self.extract_csv("failed_downloads.csv")
        self.n_failed_processing, _ = self.extract_csv("failed_processing.csv")
        self.n_failed_rendering, _ = self.extract_csv("failed_rendering.csv")
        self.n_failed_embedding_generation, _ = self.extract_csv("failed_embedding_generation.csv")
        n_amount_of_embeddings = len(os.listdir(embedding_base_path))
        statistic_dict = {
            "n_failed_downloads": self.n_failed_downloads,
            "n_failed_processing": self.n_failed_processing,
            "n_failed_rendering": self.n_failed_rendering,
            "n_failed_embedding_generation": self.n_failed_embedding_generation,
            "n_successful_embedding_generation": self.n_successful_embedding_generation,
            "n_amount_of_embeddings": n_amount_of_embeddings
        }
        return statistic_dict

    def set_statistics_by_sample(self, sample: pd.DataFrame):
        # get all sha256s from sample
        sha256s = sample.sha256.values
        self.n_samples = len(sha256s)

        # create a dataframe to store the object processing
        object_processing_df = pd.DataFrame(
            columns=["sha256", "succesful_embed", "Download_failure", "Processing_failure", "Rendering_failure", "EmbeddingClassification_failure"])
        
        embedding_sha256_names = [os.path.splitext(file_name)[0].split("_")[-1] for file_name in os.listdir(embedding_base_path)]

        # get the statistics from the logging files 
        self.n_failed_downloads, failed_downloads_sha256 = self.extract_csv("failed_downloads.csv")
        self.n_failed_processing, failed_processing_sha256 = self.extract_csv("failed_processing.csv")
        self.n_failed_rendering, failed_renderings_sha256 = self.extract_csv("failed_rendering.csv")
        self.n_failed_embedding_generation, failed_embedding_clas_sha256 = self.extract_csv("failed_embedding_generation.csv")

        for sha256 in sha256s:
            embedding_exists = sha256 in embedding_sha256_names
            download_failure = sha256 in failed_downloads_sha256
            processing_failure = sha256 in failed_processing_sha256
            rendering_failure = sha256 in failed_renderings_sha256
            embedding_generation_failure = sha256 in failed_embedding_clas_sha256
            object_processing_df.loc[len(object_processing_df)] = [sha256, embedding_exists, download_failure, processing_failure, rendering_failure, embedding_generation_failure]
        
        n_successful_downloads = len(sample) - self.n_failed_downloads
        n_successful_rendering = n_successful_downloads - \
            (self.n_failed_processing + self.n_failed_rendering)
        self.n_successful_embedding_generation = n_successful_rendering - \
            self.n_failed_embedding_generation
        self.object_processing_df = object_processing_df

    def extract_csv(self, csv_file_name: str):
        if os.path.exists(os.path.join(self.run_dir, csv_file_name)):
            csv_df = pd.read_csv(os.path.join(self.run_dir, csv_file_name))
            n_objects_in_csv = len(csv_df)
            alls_sha256_in_csv = csv_df.sha256.values
            return n_objects_in_csv, alls_sha256_in_csv
        return 0, []

def summarize_object_processing(sample: pd.DataFrame, start_index: int, end_index: int, run_dir: str, seconds_taken: int):
    summary_statistics = SummaryStatistics(run_dir)
    summary_statistics.set_statistics_by_sample(sample)
    
    # write statistics into file 
    with open(os.path.join(run_dir, "object_processing_stats.txt"), "w") as f:
        f.write(f"Time taken: {int(seconds_taken / 60)} minutes and {int(seconds_taken % 60)} seconds \n")
        f.write(str(summary_statistics))
    
    # save the object_processing_df to a csv
    summary_statistics.object_processing_df_to_csv(os.path.join(g_base_path, f"object_processing_{start_index}_{end_index}.csv"))
    return summary_statistics