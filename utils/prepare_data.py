from datasets import Dataset
from transformers import LlamaTokenizer


def prepare_sqa(problems, tokenizer):
    def tokenize(prompt):
        tknz_prompt = tokenizer(prompt, padding=False, return_tensors=None, truncation=True, max_length=1024)
        return tknz_prompt

    def generate_prompt(data_point):
        return f"""{data_point["instruction"]}
{data_point["question"]}"""

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    def convert_sqa(raw_dataset):
        converted_dataset = {}
        pid = 1
        for data_point in raw_dataset.values():
            split = data_point['split']
            if split not in converted_dataset:
                converted_dataset[split] = []

            context = None # hint
            if data_point['caption'] is not None:
                context = '\nImage: ' + data_point['caption'] + '\nContext: ' + data_point['hint']
            # ignore non-image data in the training set               
            elif split != 'train':
                context = '\nImage: ' + ' None' '\nContext: ' + data_point['hint']

            if context is not None:
                choices_with_index = [f"{index}: '{choice}'" for index, choice in enumerate(data_point['choices'])]
                question = "Main question: " + data_point['question'] + context + '\n' +f"Choices: [{', '.join(choices_with_index)}]"
                answer = str(data_point['answer'])
                formatted_data = {"instruction": "Instruction: ask two Q then select the best answer from Choices.",
                    "question": question,
                    "response": [str(pid), answer], # "response" is necessary to match PPOConfig dataloader builder
                    "explanation": data_point['solution']
                }
                converted_dataset[split].append(formatted_data)
            
            pid += 1

        return converted_dataset

    sqa = convert_sqa(problems)

    dataset_train = Dataset.from_list(sqa['train'])
    dataset_val = Dataset.from_list(sqa['val'])
    dataset_test = Dataset.from_list(sqa['test'])

    sqa_train = dataset_train.map(generate_and_tokenize_prompt)
    sqa_val = dataset_val.map(generate_and_tokenize_prompt)
    sqa_test = dataset_test.map(generate_and_tokenize_prompt)

    return sqa_train, sqa_val, sqa_test