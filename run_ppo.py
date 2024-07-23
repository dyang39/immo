from dataclasses import dataclass, field
from typing import Optional, List

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, Blip2Processor, Blip2ForConditionalGeneration
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import os
import json
import re
import warnings
from PIL import Image
from datetime import datetime

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="model/exp/warmup", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="model/vicuna-7b-1.1", metadata={"help": "the tokenizer name"})
    # reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="data/scienceqa/data", metadata={"help": "the data set path"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=12, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=10, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=1, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="model/exp/ppo/", metadata={"help": "output directory"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})

parser = HfArgumentParser(ScriptArguments)
parser.add_argument("-f", type=str, help="Path to the config file.")
parser.add_argument("--kernel-id", type=str, help="ID of the Jupyter kernel.") # for jupyter notebook usage

script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

current_device = Accelerator().local_process_index
print('Current device: ', current_device)

## load dataset
from utils.prepare_data import prepare_sqa # modified to above

tokenizer = LlamaTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # allow batched inference

problems = json.load(open(os.path.join(script_args.dataset_name, 'problems_caption_05-22-17-22.json')))

# remove all non-image data from training set
filtered_problems = {k: v for k, v in problems.items() if not (v.get('image') is None and v.get('split') == 'train')}

sqa_train, sqa_val, sqa_test = prepare_sqa(filtered_problems, tokenizer)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set seed before initializing value head for deterministic eval
set_seed(script_args.seed)

sqa_val.set_format(type="torch")
sqa_train.set_format(type="torch")
sqa_test.set_format(type="torch")

# loading activate model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
llm = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, llm.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# loading enviroment model
processor = Blip2Processor.from_pretrained("model/blip2-opt-2.7b")
vlm = Blip2ForConditionalGeneration.from_pretrained("model/blip2-opt-2.7b").to('cuda:1')

# training config
ppo_trainer = PPOTrainer(
    config,
    llm,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=sqa_train,
    data_collator=collator,
    optimizer=optimizer,
)

# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

# define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": 1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id, # vicuna_tknz.pad_token_id,
    "eos_token_id": 100_000,
}

output_min_length = 1
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

def call_vlm(vlm_prompts, pids):
    images = [Image.open(os.path.join('data/scienceqa', problems[pid]['split'], pid, 'image.png')) for pid in pids]

    processor.tokenizer.padding_side = 'left'
    inputs = processor(images, text=vlm_prompts, padding = True, return_tensors="pt").to('cuda:1')

    generated_ids = vlm.generate(**inputs, max_new_tokens=25)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [text.strip() + "." for text in generated_text]

    return (generated_text)

def call_llm(input_ids, turns):
    generated = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # all text include
    generate_ids = ppo_trainer.generate(
            input_ids,
            return_prompt=False,
            # length_sampler=output_length_sampler,
            length_sampler=LengthSampler(1, len(input_ids)+32),
            temperature = 0.1,
            **generation_kwargs,
        )
    
    generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) # all text include
    
    if turns == 1:
        q1 = []
        for x in generated:
            q1.append(re.search(r"Q1: (.+)", x).group(1) if re.search(r"Q1: (.+)", x) else "What is the image about?")            
        return q1
    
    if turns == 2:
        q2 = []
        for x in generated:
            q2.append(re.search(r"Q2: (.+)", x).group(1) if re.search(r"Q2: (.+)", x) else "Any text in the image?")
        return q2
    
    answer = []
    for x in generated:
        answer.append(re.search(r"Final answer: (\d+)", x).group(1) if re.search(r"Final answer: (\d+)", x) else '9')
    return answer


def inner_monologue(prompts_ids, pid, num_turns=2):
    '''
    prompts_ids: list of tensor in a batch
    pid: list of problem id to track the image dir
    '''
    LM_response = [''] * len(pid)
    
    question = tokenizer.batch_decode(prompts_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    conversation_history = question
            
    for turn in range(1, num_turns + 1):
        # Generate the question
        q = call_llm(prompts_ids, turn) # list of Qs
        
        LM_response = [response + f"\nQ{turn}: {sub_q}" for response, sub_q in zip(LM_response, q)]
        conversation_history = [conversation + f"\nQ{turn}: {sub_q}" for conversation, sub_q in zip(conversation_history, q)]
        
        # Call VLM
        vlm_prompt = [f"Question: {sub_q} Answer:" for sub_q in q]

        a = call_vlm(vlm_prompt, pid)
        conversation_history = [conversation + f"\nA{turn}: {sub_a}" for conversation, sub_a in zip(conversation_history, a)]
        
        # Update the prompt ids for the next iteration
        prompts_ids = []
        for conversation in conversation_history:
            prompts_ids.append(tokenizer.encode(conversation, return_tensors="pt")[0].to(device))
            
    # Final turn
    answer = call_llm(prompts_ids, "final turn")
    LM_response = [response + f"\nFinal answer: {ans}" for response, ans in zip(LM_response, answer)]
    conversation_history = [conversation + f"\nFinal answer: {sub_a}" for conversation, sub_a in zip(conversation_history, answer)]

    return LM_response, conversation_history

# ppo
warnings.filterwarnings('ignore')

for epoch in range(config.ppo_epochs):
    print("Epoch ", epoch)
    for batch in tqdm(ppo_trainer.dataloader):        
        question_tensors = (batch["input_ids"])
        pid = [response[0] for response in batch['response']] # pid, answer
        
        LM_response, conversation_history = inner_monologue(question_tensors, pid)
        
        batch["predict"] = [predict[-1] for predict in LM_response] # pid, answer
        scores = [1 if batch["predict"][i] == batch['response'][i][1] else 0 for i in range(len(batch['response']))]
        reward_baseline = 0.0 # a baseline value that is subtracted from the reward; no longer needed
        rewards = [torch.tensor(output - reward_baseline) for output in scores]

        # Run PPO step
        response_tensors = tokenizer.batch_encode_plus(LM_response, padding=True, return_tensors="pt").to(device)
        stats = ppo_trainer.step(question_tensors, list(response_tensors['input_ids']), rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")