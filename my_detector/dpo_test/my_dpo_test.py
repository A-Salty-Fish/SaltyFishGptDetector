import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = "cuda"

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

def chat(model, tokenizer, context):
    # start_time = time.time()
    messages = [
        {"role": "user", "content": context}
        # {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        # {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=512, do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    # end_time = time.time()
    # print("generate response successful: " + str(end_time - start_time))
    print(decoded[0])
    print('-----------------------')
    return decoded[0].split('[/INST]')[1].replace('</s>', '')
    # return decoded[0]

if __name__ == '__main__':
    model, tokenizer = load_test_model(perf_path='./hc3_all_1/')
    prompt = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:  I cannot definitively answer whether your chest pain is related to the intake of Clindamycin and Oxycodone without conducting a thorough examination or reviewing your medical history. However, I can share some information that may help you better understand the potential risks associated with these medications.\n\nClindamycin is an antibiotic that is sometimes associated with gastrointestinal (GI) side effects, including abdominal pain, diarrhea, and nausea. In rare cases, Clindamycin can cause serious GI conditions such as Clostridium difficile-associated diarrhea (CDAD). While chest pain is not a common side effect, it is possible that your GI symptoms could be causing referred pain in your chest.\n\nOxycodone is an opioid pain medication that is sometimes associated with side effects such as respiratory depression, dizziness, and constipation. Rarely, opioids can cause heart-related side effects such as arrhythmias or chest pain.\n\nIt is important to note that chest pain can have many different causes, including heart conditions, lung conditions, and gastroesophageal reflux disease (GERD). Therefore, it is crucial that you contact your healthcare provider as soon as possible to report your symptoms. They may recommend further testing to evaluate the underlying cause of your chest pain.\n\nIn the meantime, you can take the following steps to help manage your symptoms:\n\n1. Continue taking your medications as prescribed, but do not increase the dosage without speaking to your healthcare provider.\n2. Avoid taking large or frequent doses of opioids, as this can increase the risk of side effects.\n3. Stay hydrated by drinking plenty of water or other clear fluids.\n4. Eat smaller, more frequent meals throughout the day instead of large meals.\n5. Avoid caffeine, alcohol, and other substances that can irritate your GI tract.\n6. Practice deep breathing exercises to help reduce anxiety and improve oxygenation to your body.\n\nI hope this information is helpful. I encourage you to contact your healthcare provider with any concerns or questions you may have."
    print(chat(model, tokenizer, prompt))
