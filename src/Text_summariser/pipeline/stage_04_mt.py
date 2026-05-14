# Pipeline
from Text_summariser.config.configuration import configurationManager
from Text_summariser.components.model_trainer import ModelTrainer
from Text_summariser.logging import logger
import os


class ModelTrainerTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        model_trainer_config = config.get_model_trainer_config()
    
        model_path = os.path.join(model_trainer_config.root_dir, "t5-samsum-final")
    
        if not os.path.exists(model_path):
            print("No saved model — training now...")
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        else:
            print("Model already exists — skipping training!")



