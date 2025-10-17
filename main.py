from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PromptTuningConfig
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback
)
import torch
import evaluate
from tqdm import tqdm


# ----------------------------
# Dataset Preparation
# ----------------------------
def prepare_dataset(dataset_name: str, tokenizer, sample_train=1000, sample_valid=200):
    dataset = load_dataset(dataset_name)

    if sample_train and sample_train < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(sample_train))
    if sample_valid and sample_valid < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(sample_valid))

    # ----------------------------
    # Tokenization & label alignment
    # ----------------------------
    def preprocess(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors=None,
        )
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            answers = examples["answers"][i]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        return tokenized_examples

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        desc="Tokenizing dataset",
        load_from_cache_file=False,
    )

    return tokenized_dataset


# ----------------------------
# Evaluation
# ----------------------------
def benchmark_model(model, tokenizer, dataset):
    squad_metric = evaluate.load("squad")
    bert_metric = evaluate.load("bertscore")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    all_references = []

    tensor_columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    tensor_dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in tensor_columns]
    )

    dataloader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=8,
        collate_fn=default_data_collator,
        pin_memory=False,
    )

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch_inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        for j in range(len(start_logits)):
            start_idx = torch.argmax(start_logits[j])
            end_idx = torch.argmax(end_logits[j])

            example_index = i * dataloader.batch_size + j
            offsets = dataset[example_index]["offset_mapping"]  
            context = dataset[example_index]["context"]

            if offsets is not None and len(offsets) > end_idx:
                start_char = offsets[start_idx][0]
                end_char = offsets[end_idx][1]
                pred_text = context[start_char:end_char]
            else:
                pred_text = tokenizer.decode(
                    batch["input_ids"][j][start_idx:end_idx + 1],
                    skip_special_tokens=True,
                )
                
            example_id = dataset[example_index]["id"]

            all_predictions.append({
                "id": example_id,
                "prediction_text": pred_text,
            })

            all_references.append({
                "id": example_id,
                "answers": dataset[example_index]["answers"],
            })

    squad_scores = squad_metric.compute(predictions=all_predictions, references=all_references)
    try:
        bert = bert_metric.compute(
            predictions=[p["prediction_text"] for p in all_predictions],
            references=[a["answers"]["text"][0] for a in all_references],
            lang="en"
            )
    except Exception:
        bert = {"f1":0}


    results = {
        "F1": squad_scores["f1"],
        "ExactMatch": squad_scores["exact_match"],
        "BERTScore": sum(bert["f1"]) * 100/ len(bert["f1"]),
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f} %")

    return results



# ----------------------------
# LoRA Training Workflow
# ----------------------------
def lora_workflow(training_arguments, model_name, tokenizer, dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"], 
        bias="none",
        task_type="QUESTION_ANS",
    )

    model = get_peft_model(model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    model.disable_adapter_layers()
    benchmark_model(model, tokenizer, dataset["validation"])

    model.enable_adapter_layers()
    benchmark_model(model, tokenizer, dataset["validation"])

# ----------------------------
# Prompt Training Workflow
# ----------------------------

def prompt_workflow(training_arguments, model_name, tokenizer, dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    prompt_config = PromptTuningConfig(
        task_type="QUESTION_ANS",
        prompt_tuning_init="TEXT",  # can also be "RANDOM"
        num_virtual_tokens=20,      # typically 10-50
        prompt_tuning_init_text="Answer the question based on the context:",
        tokenizer_name_or_path=model_name,
    )

    model = get_peft_model(model, prompt_config)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    benchmark_model(model, tokenizer, dataset["validation"])


# ----------------------------
# Main Entry
# ----------------------------
def main():
    model_name = "mrm8488/bert-tiny-finetuned-squadv2"
    output_dir = "./output"
    dataset_name = "squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = prepare_dataset(dataset_name, tokenizer)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=10,
        gradient_accumulation_steps=4,
        learning_rate=3e-3,
        fp16=False,
        eval_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    lora_workflow(training_arguments, model_name, tokenizer, dataset)
    # prompt_workflow(training_arguments, model_name, tokenizer, dataset)


if __name__ == "__main__":
    main()