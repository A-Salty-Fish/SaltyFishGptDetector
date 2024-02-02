import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def load_trainer_args():
    # CUDA_VISIBLE_DEVICES=5,6 accelerate launch --multi_gpu DPO_trl.py \
    # CUDA_VISIBLE_DEVICES=5 python DPO_trl.py \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 1 \
    # --num_train_epochs 1 \
    # --save_steps 500 \
    # --save_total_limit 5 \
    # --learning_rate 5e-7 \
    # --seed 42 \
    # --ddp_find_unused_parameters=False \
    # --remove_unused_columns false \
    # --logging_steps 10 \
    # --output_dir ./weights/DPO_BC
    train_args = TrainingArguments()
    train_args.gradient_accumulation_steps = 1
    train_args.num_train_epochs = 1
    train_args.save_steps = 500
    train_args.save_total_limit = 2
    train_args.learning_rate = 5e-4
    train_args.seed = 42
    train_args.ddp_find_unused_parameters = False
    train_args.remove_unused_columns = False
    train_args.logging_steps = 100
    train_args.output_dir = './tmp'




def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    all_begin_time = time.time()

    begin_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to("cuda:0")
    print("load model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to("cuda:0")
    print("load ref_model success: " + str(time.time() - begin_time))

    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, ref_model, tokenizer


if __name__ == '__main__':
    model, ref_model, tokenizer = load_model()
    while True:
        time.sleep(5)
