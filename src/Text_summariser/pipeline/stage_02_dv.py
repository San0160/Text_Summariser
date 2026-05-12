from Text_summariser.config.configuration import configurationManager
from Text_summariser.components.data_validation import Datavalidation
from Text_summariser.logging import logger

class DatavalidationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = Datavalidation(config = data_validation_config)
        data_validation.validate_all_files_exists()