import os
from multiprocessing import Pool
from wj_preprocess_regexp_sub import Preprocess
class MultiprocessingPreprocessor:

    def __init__(self, input_dir, output_dir, num_workers=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.preprocessor = Preprocess(self.output_dir)

    def process_file(self, file_path):
        """
        Process a single file and print which worker is processing it.
        """
        worker_id = os.getpid()
        print(f"Worker {worker_id} is processing file: {file_path}")
        
        self.preprocessor(file_path)

    def process_files_in_parallel(self):
        """
        Process files in parallel using multiprocessing.
        """
        input_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]

        with Pool(processes=self.num_workers) as pool:
            pool.map(self.process_file, input_files)

    def run(self):
        """
        Entry point to start the processing.
        """
        self.process_files_in_parallel()

if __name__ == '__main__':

    input_directory = "/root/lanyun-tmp/pretrain/WanJuan1/nlp/CN"
    output_directory = "/root/lanyun-fs/team2/team2_data/token"

    processor = MultiprocessingPreprocessor(input_directory, output_directory, num_workers=4)

    processor.run()
    print("All datas are done tokenized...")
