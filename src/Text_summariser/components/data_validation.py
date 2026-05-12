
# Conponents

import os
from Text_summariser.logging import logger
from Text_summariser.entity import DatavalidationConfig

class Datavalidation:
    def __init__(self, config: DatavalidationConfig):
        self.config = config

    # simple python code that validates all files

    def validate_all_files_exists(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts", "data_injection", "samsum_dataset"))

            for file in all_files:
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"validation status: {validation_status}")
                
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e