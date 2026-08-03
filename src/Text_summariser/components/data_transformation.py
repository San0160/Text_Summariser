import os
from Text_summariser.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

class DataTransformation:

    def __init__(self, config):
        """
        Initialize the Data Transformation component.

        Args:
            config: Configuration object containing tokenizer name,
                    max input/output lengths.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def run(self):
        dataset = load_from_disk(self.config.data_path)

        tokenized_dataset = self.transform(dataset)

        tokenized_dataset.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )

    def create_prompt(self, dialogue):
        """
        Convert a dialogue into an instruction-based prompt.
        """

        return f"""Summarize the following conversation.

Dialogue:
{dialogue}

Summary:"""

    def preprocess_function(self, batch):
        """
        Tokenize the dialogue prompts and target summaries.
        """

        prompts = [
            self.create_prompt(dialogue)
            for dialogue in batch["dialogue"]
        ]

        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.MAX_INPUT_LENGTH,
            truncation=True
        )

        labels = self.tokenizer(
            text_target=batch["summary"],
            max_length=self.config.MAX_TARGET_LENGTH,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def transform(self, dataset):
        """
        Apply preprocessing to the entire dataset.
        """

        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )

        return tokenized_dataset