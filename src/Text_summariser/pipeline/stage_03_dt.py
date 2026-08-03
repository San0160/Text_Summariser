from Text_summariser.config.configuration import configurationManager
from Text_summariser.components.data_transformation import DataTransformation
from Text_summariser.logging import logger

class DataTransformationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.run()
