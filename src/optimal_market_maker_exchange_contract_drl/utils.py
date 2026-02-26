import logging
from pathlib import Path


class Logger:
    def __init__(
        self,
        save_dir: Path,
        filename: str = "training.log",
        verbose: bool = True,
        overwrite: bool = True,
    ):
        self.verbose = verbose
        self.log_path = save_dir / filename

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
