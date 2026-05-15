from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import pandas as pd
from tqdm import tqdm
import evaluate                  
rouge = evaluate.load("rouge")
from Text_summariser.entity import ModelEvaluationConfig
import os

# Conponents
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunk(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metrics_on_test_ds(self, dataset, metric, model, tokenizer,
                                      batch_size=16,
                                      device="cuda" if torch.cuda.is_available() else "cpu",
                                      column_text="dialogue",
                                      column_summary="summary"):
        
        article_batches = list(self.generate_batch_sized_chunk(dataset[column_text], batch_size))
        target_batches  = list(self.generate_batch_sized_chunk(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

            inputs = tokenizer(
                article_batch,                  # ← was article_batches (whole list!)
                max_length=1024,
                truncation=True,
                padding="max_length",           # ← was max_lenght
                return_tensors="pt"             # ← was return_tensor
            )

            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),        # ← was input not inputs
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,             # ← was lenght_penalty
                num_beams=8,
                max_length=128
            )

            decoded_summaries = [
                tokenizer.decode(s,
                    skip_special_tokens=True,           # ← was skip_special_token
                    clean_up_tokenization_spaces=True)  # ← was clean_up_tokenization_sapce
                for s in summaries
            ]

            decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]  # ← was replace("", " ")

            metric.add_batch(                   # ← was metrics (no s)
                predictions=decoded_summaries,  # ← was decoded_summariess and prediction
                references=target_batch         # ← was target_bactch and referenmce
            )

        score = metric.compute()
        return score

    def evaluate(self):                         # ← was evaulate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)        # ← was frompretrainede
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)  # ← was Automodelforseq2seqlm

        # Load rouge — new way!
        rouge = evaluate.load("rouge")          # ← replaces load_metric

        # Load data
        dataset_samsum = load_from_disk(self.config.data_path)

        # Run evaluation
        score = self.calculate_metrics_on_test_ds(
            dataset=dataset_samsum["test"],
            metric=rouge,
            model=model,
            tokenizer=tokenizer,
            batch_size=16,
            device=device,
            column_text="dialogue",             # ← SAMSum column names
            column_summary="summary"
        )

        # Create the correct directory
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)

        # Save metrics
        df = pd.DataFrame([score], index=["rouge_score"])

        df.to_csv(
            self.config.metric_file_name,
            index=False
        )

        print("Evaluation scores:", score)