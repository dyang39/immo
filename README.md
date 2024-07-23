# IMMO: Inner Monologue Multi-modal Optimization 
[Tackling Vision Language Tasks Through Learning Inner Monologues](https://arxiv.org/abs/2308.09970) 
The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024

## Usage
This repo is still under construction. More content will be updated soon. 

### 1. Set up environment
```pip install -r requirements.txt```

[Alternatively] setup docker image using ```dockerfile```

### 2. Download dataset
Follow the offcial instruction to download [ScienceQA](https://scienceqa.github.io/) and [A-OKVQA](https://allenai.org/project/a-okvqa/home) datasets to ```data/```.

### 3. Supervised fine-tuning
- Example usage:
```python3 run_ppo.py --base_model="lmsys/vicuna-7b-v1.2" --output_dir="model/exp/warmup/" --data_path = 'data/aokvqa'```

### 4. Reinforcement learning
- Example usage:
```python3 run_warmup.py --model_name="model/exp/warmup" --output_dir="model/exp/rl" --data_path ='data/scienceqa'```

## Related code base
We thank the [huggingface trl](https://github.com/huggingface/trl/tree/main) for providing the engine and authors of [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) for releasing the code base.

## Note:
- The checkpoints from sl-warmup can be downloaded [here](https://drive.google.com/file/d/1VTTM9wNjKSyXRzc1zfY4odK23etRBcLC/view?usp=sharing). Note this implementation is based on Llama-7b, and a stronger performance could be expected if we re-train everything from Llama-2 family.
