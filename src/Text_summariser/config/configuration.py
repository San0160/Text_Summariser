from Text_summariser.constant import *
from Text_summariser.utils.common import read_yaml, create_directories
from Text_summariser.entity import (DatainjectionConfig, DatavalidationConfig, DataTransformationConfig)


class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath) 

        create_directories([self.config.artifacts_root])# same upto here for most pipeline

    def get_data_injection_config(self) -> DatainjectionConfig:
        config = self.config.data_injection

        create_directories([config.root_dir])

        data_injection_config = DatainjectionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_injection_config
    
    def get_data_validation_config(self) -> DatavalidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DatavalidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            ALL_REQUIRED_FILES = config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name
        )

        return data_transformation_config