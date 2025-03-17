import os
import fcntl
from multiprocessing import Process
import multiprocessing
from multiprocessing.queues import Queue
import argparse
import objaverse.xl as oxl
import objaverse
import pandas as pd
from typing import Any, Dict, Hashable, List
import shutil
import logging
import uuid
from tqdm import tqdm
import wandb
import time
from setting import g_device, g_base_path, g_objaverse_base_path, g_num_renders, g_max_batch_threshold, g_keep_rendered_objects
from oxl_embedder.status_logging import embedding_base_path, failed_download_folder, found_objects_folder
from oxl_embedder.status_logging import init_logging, create_logging_files, process_download_and_processing_failure, SummaryStatistics, summarize_object_processing, log_object_to_csv, check_if_sha256_in_csv
from oxl_embedder.processing_renders import RenderedImageProcessor, ClassificationQueueHandler
from oxl_embedder.loaded_object_handler import LoadedObjectHandler
# objaverse_base_path = "/Users/damian/.objaverse"
objaverse._VERSIONED_PATH = g_objaverse_base_path

class_estimation_csv_name = "car_model_class_estimation.csv"
object_detection_csv_name = "car_model_object_detection.csv"
lock_file = os.path.join(g_base_path, "lock_file.lock")
render_processor = None

def worker_init(max_batch_threshold: int = g_max_batch_threshold, run_dir: str = g_base_path):
    """Initialize models once per worker."""
    class_estimation_csv = os.path.join(run_dir, class_estimation_csv_name)
    if not os.path.exists(class_estimation_csv):
        with open(class_estimation_csv, "w") as f:
            f.write("object_uid,dino_class,siglip_class\n")
    object_detection_csv = os.path.join(run_dir, object_detection_csv_name)
    if not os.path.exists(object_detection_csv):
        with open(object_detection_csv, "w") as f:
            f.write("object_uid,yolo_labels\n")
    global render_processor
    render_processor = ClassificationQueueHandler(base_folder=embedding_base_path, class_estimation_csv=class_estimation_csv, 
                                                  object_detection_csv=object_detection_csv, expected_sequence_length=g_num_renders, 
                                                  device=g_device, run_dir=run_dir,
                                                  max_batch_threshold=max_batch_threshold, keep_rendered_objects=g_keep_rendered_objects)

def acquire_lock():
    """Acquire a lock on the lock file."""
    lock_fd = open(lock_file, "w")
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd  # Lock acquired
        except IOError:
            print("Waiting for lock...")
            time.sleep(5)

def release_lock(lock_fd):
    """Release the lock on the lock file."""
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

class RenderingQueue(Queue):
    """A queue for managing rendering tasks with GPU batching.

    This class extends the multiprocessing Queue to handle rendering tasks with GPU batching.
    It maintains a run directory for output and configurable GPU batch sizes.

    Attributes:
        run_dir (str): Directory path for the current rendering run outputs
        gpu_batch_size (int): Number of items to process in each GPU batch, defaults to 20
    """

    def __init__(self, *args, **kwargs):
        # for some reason the ctx is not passed correctly if we do it not explicitly
        ctx = kwargs.pop('ctx', multiprocessing.get_context())
        # Initialize the Queue
        super().__init__(ctx=ctx, *args, **kwargs)
        self.run_dir = None
        self.gpu_batch_size = 20

    def set_run_dir(self, run_dir: str):
        self.run_dir = run_dir

    def set_gpu_batch_size(self, gpu_batch_size: int):
        self.gpu_batch_size = gpu_batch_size

    def get_run_dir(self):
        return self.run_dir
    
    def get_gpu_batch_size(self):
        return self.gpu_batch_size

def embed_and_classify_objects(queue):
    """
    Process chunks of objects from a queue for embedding and classification.
    This function continuously pulls chunks of objects from a queue and processes them
    for embedding and classification. If an error occurs during processing, it will
    attempt to recover by reinitializing with a smaller batch size.
    """
    while True:
        chunk = queue.get()
        run_dir = queue.get_run_dir()
        gpu_batch_size = queue.get_gpu_batch_size()
        if chunk is None:  # Sentinel to break loop
            global render_processor
            if render_processor is not None:
                render_processor.empty_pending_classifications()
            break
        try:
            process_renders(chunk, run_dir, gpu_batch_size)  # Embed and classify the chunk
        except Exception as e:
            # reset the render_processor and try with smaller batch size 
            logging.error(f"Error while processing renders: {e}")
            # create a txt file to log the error, with the current time
            error_file_name = f"error_{time.time()}.txt"
            with open(os.path.join(run_dir, error_file_name), "w") as f:
                f.write(f"Error while processing renders: {e}")
            del render_processor
            render_processor = None
            worker_init(max_batch_threshold=gpu_batch_size // 2, run_dir=run_dir)
            process_renders(chunk, run_dir, gpu_batch_size // 2)

def process_renders(chunk: pd.DataFrame, run_dir: str, gpu_batch_size: int):
    """Process a chunk of rendered 3D objects for classification.
    This function processes a batch of rendered 3D objects by:
    1. Initializing the render processor if not already initialized
    2. Finding and processing render files for each object in the chunk
    3. Handling failed renders and downloads appropriately
    4. Processing any pending classifications
    """
    # initialize render_processor
    global render_processor
    if render_processor is None:
        worker_init(max_batch_threshold=gpu_batch_size, run_dir=run_dir)

    # go through the chunk and add the renders to the render_processor
    render_base_path = os.path.join(g_base_path, "renders")
    for index, row in chunk.iterrows():
        sha256 = row["sha256"]
        render_file_path = os.path.join(render_base_path, sha256)
        found_render = False
        # check if render_file_path exists
        if os.path.exists(render_file_path):
            # check if the folder has a zip file
            zip_file_paths = [os.path.join(render_file_path, file_name) for file_name in os.listdir(render_file_path) if file_name.endswith(".zip")]
            if len(zip_file_paths) == 1:
                render_processor.add_classification(render_file_path)
                found_render = True
        if not found_render:
            # check if the file was downloaded, if yes the rendering failed     
            sha256_string = f"{sha256}.csv"
            if os.path.isfile(os.path.join(found_objects_folder, sha256_string)):
                # check if processing failed
                if not check_if_sha256_in_csv(os.path.join(run_dir, "failed_processing.csv"), sha256):
                    log_object_to_csv(os.path.join(run_dir, "failed_rendering.csv"), sha256, row["fileIdentifier"])
                # remove the file from the found objects folder
                os.remove(os.path.join(found_objects_folder, sha256_string))
            else:
                # if the file was not logged as downloaded, the download failed
                log_object_to_csv(os.path.join(run_dir, "failed_downloads.csv"), sha256, row["fileIdentifier"])

    # process pending classifications
    render_processor_loop_counter = 0
    while render_processor.get_pending_classifications() > 0:
        render_processor.process_pending_classifications()
        render_processor_loop_counter += 1
        if render_processor_loop_counter > 10:
            raise Exception("Iterated more than 10 times while waiting for pending classifications")

def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    # Check if local_path exists and is a file
    if not os.path.isfile(local_path):
        sha256_string = f"{sha256}.csv"
        failed_file_to_log = os.path.join(failed_download_folder, sha256_string)
        log_object_to_csv(failed_file_to_log, sha256, file_identifier)
        return
    
    sha256_string = f"{sha256}.csv"
    found_file_to_log = os.path.join(found_objects_folder, sha256_string)
    log_object_to_csv(found_file_to_log, sha256, file_identifier)
    
    # move the file to the shape folder
    filename = os.path.basename(local_path)
    shape_path = os.path.join(g_base_path, "shape")
    os.makedirs(shape_path, exist_ok=True)
    new_path = os.path.join(shape_path, sha256)
    os.makedirs(new_path, exist_ok=True)
    new_file_name = f"{sha256}.{filename.split('.')[-1]}"
    new_file_path = os.path.join(new_path, new_file_name)
    shutil.copy2(local_path, new_file_path)

    object_loader = LoadedObjectHandler(sha256, file_identifier)
    object_loader.render_object()

def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any]
) -> None:
    sha256_string = f"{sha256}.csv"
    failed_file_to_log = os.path.join(failed_download_folder, sha256_string)
    log_object_to_csv(failed_file_to_log, sha256, file_identifier)
    return

def get_global_start_index(sample_size: int, continue_from_crashed: bool):
    # create a file to safe the current chunk
    # this file keeps global track of the current downloading state
    # so we can start multiple runs in parallel each starting from the last index of the most recent process
    current_chunk_file_path = os.path.join(g_base_path, "global_processing_state.csv")
    if not os.path.exists(current_chunk_file_path):
        sample_start = 0
        sample_end = sample_start + sample_size
        with open(current_chunk_file_path, "w") as f:
            f.write(f"{sample_end}\n")
    else:
        # read last line in csv to set the start index
        with open(current_chunk_file_path, "r") as f:
            lines = f.readlines()
            sample_start = int(lines[-1])
        if continue_from_crashed:
            # find the folder which ends with sample_start
            run_folders = [folder for folder in os.listdir(g_base_path) if folder.endswith(str(sample_start))]
            if len(run_folders) == 0:
                raise Exception(f"No run folder found for sample_start: {sample_start}")
            last_run_dir = os.path.join(g_base_path, run_folders[-1])
            last_run_current_chunk_file_path = os.path.join(last_run_dir, "current_chunk_state.csv")
            with open(last_run_current_chunk_file_path, "r") as f:
                lines = f.readlines()
                sample_start = int(lines[-1])
        sample_end = sample_start + sample_size
        with open(current_chunk_file_path, "a") as f:
            f.write(f"{sample_end}\n")
    return sample_start, sample_end

def main(sample_size: int, n_processes: int, batch_size: int = 10, continue_from_crashed: bool = False, start_index: int = None, gpu_batch_size: int = 20):
    if not os.path.exists(g_base_path):
        os.makedirs(g_base_path, exist_ok=True)
    
    lock_fd = acquire_lock()
    try:
        init_logging()

        # load the annotations
        my_annotations_to_load = oxl.get_alignment_annotations(
            download_dir=g_objaverse_base_path  # default download directory
        )
        
        # initialize the index using the global index file or the start index   
        if start_index is None:
            # first let's get the current index to start from
            sample_start, sample_end = get_global_start_index(sample_size, continue_from_crashed)

            if sample_start >= len(my_annotations_to_load):
                raise Exception(f"Sample start index {sample_start} is greater than the number of objects {len(my_annotations_to_load)}, exiting.")
            if len(my_annotations_to_load) < sample_end:
                sample_end = len(my_annotations_to_load)

            # let's generate a specific run directory so that we can keep track of the runs even for parallel runs
            run_dir = os.path.join(g_base_path, f"run_{sample_start}_{sample_end}")
            if os.path.exists(run_dir):
                print(f"Run directory {run_dir} already exists. This will overwrite the existing directory.")
            os.makedirs(run_dir, exist_ok=True)
        else:
            sample_start = start_index
            sample_end = sample_start + sample_size
            run_dir = os.path.join(g_base_path, f"run_{sample_start}_{sample_end}")
            os.makedirs(run_dir, exist_ok=True)
    finally:
        release_lock(lock_fd)
    
    # init wandb to track progress
    run = wandb.init(project="Objaverse-xl-processing",
                     config = {"sample_start": sample_start, "sample_end": sample_end, 
                     "sample_size": sample_size, "n_processes": n_processes, 
                     "batch_size": batch_size, "continue_from_crashed": continue_from_crashed,
                     "run_dir": run_dir})

    # get the explizit samples we want - from start to end index
    sample = my_annotations_to_load[sample_start:sample_end]

    # create a unique id for this run to have a file which saves the information which we can download an check 
    run_id = str(uuid.uuid4())
    file_download_csv = g_base_path + f"file_download_{run_id}.csv"
    sample.to_csv(file_download_csv, index=False)

    run_current_chunk_file_path = os.path.join(run_dir, "current_chunk_state.csv")
    with open(run_current_chunk_file_path, "w") as f:
        f.write(f"{sample_start}\n")

    # create logging files
    create_logging_files(run_dir)

    # split sample in chunks of 10
    if sample_size > batch_size:
        div_by_n_objects = sample_size - (sample_size % batch_size)
        chunks = [sample[i:i+batch_size] for i in range(0, div_by_n_objects, batch_size)]
        # add the remaining objects as the last chunk
        chunks.append(sample[div_by_n_objects:])
    else:
        chunks = [sample]

    # create a queue for the rendered objects to be processed in parallel 
    classification_queue = RenderingQueue()
    classification_queue.set_run_dir(run_dir)
    classification_process = Process(
        target=embed_and_classify_objects, args=(classification_queue,))
    classification_process.start()
    
    time_start = time.time()
    current_chunk_index = 0
    for chunk in tqdm(chunks):
        oxl.download_objects(
            objects=chunk,
            processes=n_processes,
            handle_found_object=handle_found_object,
            handle_missing_object=handle_missing_object
        )
        process_download_and_processing_failure(chunk, run_dir)
        classification_queue.put(chunk)  # Send chunk to classifier
        # update the current chunk file
        current_chunk_index += 1
        current_chunk_index_times_batch_size = current_chunk_index * batch_size + sample_start
        if current_chunk_index_times_batch_size > sample_size + sample_start:
            current_chunk_index_times_batch_size = sample_size + sample_start 
        run.log({"Current Chunk Index": current_chunk_index_times_batch_size})
        summary_statistics = SummaryStatistics(run_dir)
        statistic_dict = summary_statistics.update_download_rendering_statistics()
        run.log(statistic_dict)
        with open(run_current_chunk_file_path, "a") as f:
            f.write(f"{current_chunk_index_times_batch_size}\n")

    # empty the classification queue
    classification_queue.put(None)  # Sentinel
    classification_process.join() # wait for the process to finish
    time_end = time.time()
    seconds_taken = time_end - time_start
    print(f"Time taken: {int(seconds_taken / 60)} minutes and {int(seconds_taken % 60)} seconds")
    
    # summarize the object processing
    summary_statistics = summarize_object_processing(sample, sample_start, sample_start + sample_size, run_dir, seconds_taken)
    statistic_dict = summary_statistics.update_download_rendering_statistics()
    run.log(statistic_dict)
    run.log_code()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=10, help="Number of objects to download and process")
    parser.add_argument("--start_index", type=int, default=None, help="By default we start at zero and by each call we continue where we stopped. If set: Start index to start from, we force to start from this index, even if we overwrite the global state")
    parser.add_argument("--n_processes", type=int, default=8, help="Number of processes to use for downloading and rendering the objects")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of objects to Download and render before we pass them to the embedding models to be processed as a batch") 
    parser.add_argument("--continue_from_crashed", action="store_true", help="If the last run crashed, we can continue from the last index which was successfully processed")
    parser.add_argument("--gpu_batch_size", type=int, default=20, help="Number of objects to process in parallel on the GPU")
    args = parser.parse_args()

    main(args.sample_size, args.n_processes, args.batch_size, args.continue_from_crashed, args.start_index, args.gpu_batch_size)
