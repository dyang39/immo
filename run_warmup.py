import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

'''
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
'''

import json
import re

def train(
    # model/data params
    base_model: str = "lmsys/vicuna-7b-v1.5", # default to vicuna 1.5
    output_dir: str = "checkpoints/warmup/",
    data_path = 'data/aokvqa',
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 15,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 2000,

    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    add_eos_token: bool = False,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):

    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"output_dir: {output_dir}\n"
        f"output_dir: {data_path}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # load data
    train = json.load(open(os.path.join(data_path, 'aokvqa_v1p0_train.json')))

    # load model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
    )

    try:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    except:
        print("No Lora is applied. Full model will be finetuned.")

    # tokenize
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_prompt(data_point):
        return f"""{data_point["Instruction"]}
    {data_point["Question"]}
    {data_point["Answer"]}"""

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    def convert_okvqa(raw_dataset):
        generated_qa = json.load(open(os.path.join(data_path, 'rationale/qa_rationale_gpt.json.json')))
        image_caption = json.load(open(os.path.join(data_path, 'blip2_caption.json')))
        converted_dataset = {}
        for i in range(len(generated_qa)):
            split = 'train'
            if split not in converted_dataset:
                converted_dataset[split] = []

            choices_with_index = [f"{index}: '{choice}'" for index, choice in enumerate(raw_dataset[i]['choices'])]
            image_text = image_caption[i]
            question = "Main question: " + raw_dataset[i]['question'] + '\nImage: ' + image_text + '\nContext: None\n' +f"Choices: [{', '.join(choices_with_index)}]" + '\n'

            # agent_qa
            try:
                agent_qa = re.search(r"(.*?)(?=\nReason)", generated_qa[i], re.DOTALL).group(1)
            except:
                agent_qa = generated_qa[i]
            agent_qa = agent_qa.rstrip(" \n")

            # rationale
            rationale = "\nExplaination: " + max(train[i]['rationales'], key=len) # select the longest rationales

            answer = raw_dataset[i]['correct_choice_idx']
            formatted_data = {
                "Instruction": "Instruction: ask two Q then select the best answer from Choices.",
                "Question": question + agent_qa,
                "Answer": "Final answer: " + str(answer) + rationale,
            }
            converted_dataset[split].append(formatted_data)

        return converted_dataset

    problems1 = convert_okvqa(train)
    dataset_train = Dataset.from_list(problems1['train'])
    train_data = dataset_train.map(generate_and_tokenize_prompt)

    print("train data: ", train_data)

    # train
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)