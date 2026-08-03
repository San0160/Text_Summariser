from Text_summariser.config.configuration import configurationManager
from transformers import AutoTokenizer, pipeline


class PredictionPipeline():

    def __init__(self):

        self.config = configurationManager().get_model_evaluation_config()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        self.pipe = pipeline("summarization", model=self.config.model_path, tokenizer=self.tokenizer)

    def predict(self, text):
        
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128
        }
     
        output = self.pipe(text, **gen_kwargs)

        return output[0]["summary_text"]