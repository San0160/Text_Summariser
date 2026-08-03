# Pipeline
from Text_summariser.config.configuration import configurationManager
from Text_summariser.components.model_evaluation import ModelEvaluation
from Text_summariser.logging import logger
import os


class ModelEvaluationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
    
        if not os.path.exists(model_evaluation_config.metric_file_name):
            logger.info("No metrics found — running evaluation...")
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        else:
            logger.info("Metrics already exist — skipping evaluation!")