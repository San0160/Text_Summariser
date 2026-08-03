from Text_summariser.constant import *
from Text_summariser.utils.common import read_yaml, create_directories
from Text_summariser.entity import (DatainjectionConfig, DatavalidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)


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
        params = self.params.transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name,
            MAX_INPUT_LENGTH = params.MAX_INPUT_LENGTH,
            MAX_TARGET_LENGTH = params.MAX_TARGET_LENGTH
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config= ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_ckpt = config.model_ckpt,
            model_path = config.model_path,
            model_URL = config.model_URL,
            num_train_epochs = params.num_train_epochs,
            warmup_steps = params.warmup_steps,
            per_device_train_batch_size = params.per_device_train_batch_size,
            per_device_eval_batch_size = params.per_device_eval_batch_size,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            save_strategy = params.save_strategy,
            gradient_accumulation_steps = params.gradient_accumulation_steps,
            load_best_model_at_end = params.load_best_model_at_end,
            predict_with_generate = params.predict_with_generate,
            fp16 = params.fp16
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(  # ← Capital M and E and C
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name
        )

        return model_evaluation_config