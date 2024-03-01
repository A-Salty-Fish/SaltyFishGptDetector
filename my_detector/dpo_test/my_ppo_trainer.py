import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from trl import PPOConfig, AutoModelForCausalLMWithValueHead

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from trl import  PPOConfig, PPOTrainer


# bnb_config = BitsAndBytesConfig(
#     # load_in_8bit=True,
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)



class MyClassifier(nn.Module):
    def __init__(self, base_model):
        super(MyClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


def init_model_and_tokenizer(ppo_config):
    # model = AutoModelForCausalLMWithValueHead.from_pretrained(
    #     config.model_name,
    #     # is_trainable=True,
    #     low_cpu_mem_usage=True,
    #     quantization_config = bnb_config,
    #     # torch_dtype=torch.float16,
    #     trust_remote_code=True,
    #     device_map={"": 0}
    # ).cuda()
    peft_config = LoraConfig(
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        # is_trainable=True,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        # torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
        peft_config=peft_config
    )
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


from transformers import pipeline


def init_reward_model(reward_model_name, reward_model_path=None):
    if reward_model_name == "lvwerra/distilbert-imdb":
        reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")
        return reward_model
    elif reward_model_name == "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = (torch.load(reward_model_path))
        model.eval()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        model = model.to(device)

        def reward_func(texts):
            results_predictions = []
            data_inputs = []
            for text in texts:
                data_inputs.append(
                    tokenizer(text, padding='max_length',
                              max_length=512,
                              truncation=True,
                              return_tensors="pt")
                )

            with torch.no_grad():
                model.eval()
                for data_input in data_inputs:
                    # print(data_input)
                    attention_mask = data_input['attention_mask'].to(device)
                    input_ids = data_input['input_ids'].squeeze(1).to(device)

                    output = model(input_ids, attention_mask)

                    output = (output > 0.5).int()
                    results_predictions.append(output)

            npmpy_results = torch.cat(results_predictions).cpu().detach().numpy()
            return [x[0] * 1.0 for x in npmpy_results]

        return reward_func


def init_reward_func(reward_model_name, reward_model):
    if reward_model_name == 'lvwerra/distilbert-imdb':
        def reward_func(texts):
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            return rewards
    elif reward_model_name == "roberta-base":
        return reward_model
    else:
        def reward_func(texts):
            return [0.0 for _ in texts]

    return reward_func


def tokenize_row_data(tokenizer, sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


class MyPromptDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_nums=None):
        with open(train_file, 'r', encoding='utf-8') as in_f:
            datas = json.load(in_f)
        if max_nums is None:
            self.queries = [x['prompt'] for x in datas]
            self.input_ids_list = [tokenizer.encode(query) for query in tqdm(self.queries)]
        else:
            self.queries = [x['prompt'] for x in datas][0: max_nums]
            self.input_ids_list = [tokenizer.encode(query) for query in tqdm(self.queries)][0: max_nums]

    def __getitem__(self, idx):
        # text = self.texts[idx]
        # label = self.labels[idx]
        input_ids = self.input_ids_list[idx]
        query = self.queries[idx]

        data = {}
        data['input_ids'] = input_ids
        data['query'] = query
        return data

    def __len__(self):
        return min(len(self.queries), len(self.input_ids_list))


from trl import PPOTrainer


def begin_train_ppo(
        ppo_trainer,
        tokenizer,
        save_model_path,
        reward_func,
        generation_kwargs,
        device='cuda:0'
):
    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataset):
            # print(batch)
            query_tensors = torch.tensor(batch['input_ids']).to(device)
            # query_tensors = [torch.tensor(x).to(device) for x in batch['input_ids']]
            # print(query_tensors)
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs).to(device)
            # response_tensors = [ppo_trainer.generate(x, **generation_kwargs) for x in query_tensors]
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute reward score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]

            # print("texts")
            # print(texts)
            # pipe_outputs = reward_model(texts)
            # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            rewards = [torch.tensor(x).to(device) for x in reward_func(texts)]

            response_tensors = response_tensors[0].to(device)

            # print(query_tensors)
            # print(response_tensors)

            #### Run PPO step
            # print([query_tensors])
            # print([response_tensors])
            # print(rewards)
            stats = ppo_trainer.step([query_tensors], [response_tensors], rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    ppo_trainer.save_pretrained(save_model_path)
    final_output_path = os.path.join(save_model_path, "final_checkpoint")
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    ppo_trainer.model.save_pretrained(final_output_path)


if __name__ == '__main__':
    reward_model = init_reward_model("roberta-base", 'hc3_row.pt')
    reward_func = init_reward_func("roberta-base", reward_model)

    # text1 = 'To address the challenges of data reliability in NAND flash storage, a new decoding algorithm called variable-node-based belief-propagation with message pre-processing (VNBP-MP) has been proposed for binary low-density parity-check (LDPC) codes. The algorithm utilizes the unique characteristics of the NAND flash channel to perform message pre-processing (MP), which effectively prevents the spread of unreliable messages and speeds up the propagation of reliable messages. Additionally, the VNBP-MP algorithm includes a treatment for oscillating variable nodes (VNs) to further accelerate decoding convergence. \n\nSimulation results demonstrate that the proposed VNBP-MP algorithm has significantly improved convergence speed without compromising error-correction performance, in comparison to existing algorithms.", "label": 1}, {"content": "Many techniques have been proposed to solve the simultaneous localization and mapping (SLAM) problem, and among them, the Particle Filter (PF) is considered to be one of the most effective ways. However, the PF algorithm requires a large number of samples to approximate the posterior probability density of the system, which makes the algorithm complex. Furthermore, the judgment of resampling is imperfect. In light of these challenges, this paper proposes an improved PF algorithm that introduces a population diversity factor and a genetic algorithm into the process of resampling. \n\nThe improved PF algorithm uses the effective sample size and the population diversity factor to determine whether to resample the particle set. When resampling is needed, the genetic algorithm is utilized to optimize the particle set. The simulation results demonstrate that the estimation accuracy of the improved algorithm is superior to that of the traditional particle filter, not only in terms of accuracy but also in efficiency.'
    # text2 = ' Microsoft Translator is a cloud-based translATION service provided by Microsoft. It enables real-time and automated translations between various languages. This service can be integrated into various applications, websites, and services, allowing users to translate text, spoken language, and even entire documents with ease. Microsoft Translator uses advanced artificial intelligence and machine learning technologies to provide accurate translations. It supports a wide range of languages, including but not limited to, English, Spanish, French, Chinese, German, Italian, Russian, and Japanese. Microsoft Translator is often used for international communication, multilingual websites, and global businesses to help break down language barriers.'
    # print(reward_func([text1, text2]))

    ppo_config = PPOConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        learning_rate=1e-5,
        batch_size=1,
        ppo_epochs=5,
    )
    model, tokenizer = init_model_and_tokenizer(ppo_config)

    train_file = './data/1.adv.1000.1.train'
    train_dataset = MyPromptDataset(train_file, tokenizer)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: x,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        'max_new_tokens': 512,
    }

    save_model_path = './ppo_1'

    begin_train_ppo(
        ppo_trainer,
        tokenizer,
        save_model_path,
        reward_func,
        generation_kwargs
    )

    pass
