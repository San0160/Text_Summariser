# Conponents
import os
from Text_summariser.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, examples):
        inputs = ["summarize: " + doc for doc in examples["dialogue"]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        labels = self.tokenizer(
            text_target=examples["summary"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
    
    def convert(self):
        # Load dataset from disk
        dataset_samsum = load_from_disk(self.config.data_path)

        # Tokenize dataset
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)