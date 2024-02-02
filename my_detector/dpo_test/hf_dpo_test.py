from trl import DPOTrainer


def return_prompt_and_responses(samples) -> Dict[str, str, str]:
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
            for question in samples["question"]
        ],
        "chosen": samples["response_j"], # rated better than k
        "rejected": samples["response_k"], # rated worse than j
    }

dataset = load_dataset(
    "lvwerra/stack-exchange-paired",
    split="train",
    data_dir="data/rl"
)
original_columns = dataset.column_names

dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)

dpo_trainer = DPOTrainer(
    model, # 经 SFT 的基础模型
    model_ref, # 一般为经 SFT 的基础模型的一个拷贝
    beta=0.1, # DPO 的温度超参
    train_dataset=dataset, # 上文准备好的数据集
    tokenizer=tokenizer, # 分词器
    args=training_args, # 训练参数，如: batch size, 学习率等
)

dpo_trainer.train()