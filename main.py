from Text_summariser.pipeline.stage_01_di import DataInjectionTrainingPipeline
from Text_summariser.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_injestion = DataInjectionTrainingPipeline()
    data_injestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e