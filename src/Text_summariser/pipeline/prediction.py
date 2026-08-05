from Text_summariser.config.configuration import configurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class PredictionPipeline:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       

        self.config = configurationManager().get_model_evaluation_config()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)
            
    def predict(self, text):

        if not text.strip():
            return "Enter chats or text."

        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )

        summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary