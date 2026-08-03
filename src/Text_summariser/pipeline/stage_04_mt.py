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
    
        model_path = os.path.join(
            model_trainer_config.model_path,
            "working_model"
        )
    
        model_trainer = ModelTrainer(config=model_trainer_config)
    
        if os.path.exists(model_path):
            print("Model already exists — skipping training!")
    
        else:
            print("Model not found.")
    
            try:
                model_trainer.download_model()
    
                print("Model downloaded successfully.")
    
            except Exception:
                raise RuntimeError(
                    "\nModel download failed.\n"
                    "Please ask the administrator to update the ModelTrainer code."
                )



