import logging
import os


class Logger:
    def __init__(self, save_dir, verbose=True, filename="training.log", overwrite=True):
        self.verbose = verbose
        self.log_path = os.path.join(save_dir, filename)

        os.makedirs(save_dir, exist_ok=True)

        # Determine file mode based on overwrite parameter
        filemode = "w" if overwrite else "a"

        logging.basicConfig(
            filename=self.log_path,
            filemode=filemode,
            level=logging.INFO,
            format="%(message)s",
        )
        self.logger = logging.getLogger()

    def log(self, msg):
        if self.verbose:
            print(msg)
        if self.logger is not None:
            self.logger.info(msg)
