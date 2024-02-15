import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_test_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", perf_path='./tmp.pt/checkpoint-1600'):
    all_begin_time = time.time()


    model = AutoModelForCausalLM.from_pretrained(perf_path,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=True)
    print("load model success: " + str(time.time() - all_begin_time))


    begin_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("load tokenizer success: " + str(time.time() - begin_time))

    print("load all success: " + str(time.time() - all_begin_time))
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_test_model()
