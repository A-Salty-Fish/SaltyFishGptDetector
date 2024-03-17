import json
import os
import time

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, load_in_4bit=True, low_cpu_mem_usage=True)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        # self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


def paraphase(dp, input_text, lex_diversity=20):
    prompt = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:"
    output_l60_sample = dp.paraphrase(input_text, lex_diversity=lex_diversity, order_diversity=0, prefix=prompt, do_sample=True,
                                      top_p=0.75, top_k=None, max_length=512)

    return output_l60_sample

def generate_datas(dp, output_dir='dp', file_pre='test'):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    base_dir = '../../my_detector/roberta_test/data/'
    files = [
        # 'hc3_row.train',
        # 'hc3_row.test',
        'cheat_generation.test',
        # 'cheat_polish.test',
        'ghostbuster_claude.test',
        'hc3_plus_qa_row.test',
        # 'open_qa.academic.test',
        # 'open_qa.continue.test',
        # 'open_qa.difficult.test',
        # 'open_qa.easy.test',
        # 'open_qa.test',
        # 'open_qa.rewrite.test',
        'reddit_chatGPT.test',
        # 'reddit_cohere.test',
        # 'reddit_davinci.test',
        # 'reddit_dolly.test',
        # 'reddit_flant5.test',
        'wikipedia_chatgpt.test',
        # 'wikipedia_cohere.test',
        # 'wikipedia_davinci.test',
        # 'wikipedia_dolly.test',
    ]
    for file in files:
        generate_paraphase_data(base_dir + file, file, dp, output_dir, file_pre)


def generate_paraphase_data(file_path, file_name, dp, output_dir='./qwen/', file_pre='', max_num=1000):
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: "
    print(file_path)
    try:
        with open(output_dir + file_name + file_pre + '.jsonl', 'r', encoding='utf-8') as out_f:
            existed_lines = 0
            for line in out_f:
                existed_lines += 1
            print(existed_lines)
    except Exception as e:
        existed_lines = 0
    with open(file_path, 'r', encoding='utf-8') as in_f:
        cur = 0
        with open(output_dir + file_name + file_pre + '.jsonl', 'a', encoding='utf-8') as out_f:

            file_objs = json.load(in_f)
            ai_objs = [x for x in file_objs if x['label'] == 1][0: max_num]
            results = []

            for ai_obj in tqdm(ai_objs):
                cur += 1
                if cur < existed_lines:
                    continue
                ai_content = ai_obj['content']
                ai_rewrite = paraphase(dp, ai_obj['content'], 100)
                new_ai_obj = {
                    'label': 1,
                    'prompt': prompt_template + ai_obj['content'],
                    'ai': ai_content,
                    'ai_rewrite': ai_rewrite
                }
                out_f.write(json.dumps(new_ai_obj) + '\n')


def mix_rewrite_and_human_data(dir, nums=1000):
    base_dir = '../../my_detector/roberta_test/data/'
    files = [
        # 'hc3_row.train',
        # 'hc3_row.test',
        'cheat_generation.test',
        # 'cheat_polish.test',
        'ghostbuster_claude.test',
        'hc3_plus_qa_row.test',
        # 'open_qa.academic.test',
        # 'open_qa.continue.test',
        # 'open_qa.difficult.test',
        # 'open_qa.easy.test',
        # 'open_qa.test',
        # 'open_qa.rewrite.test',
        'reddit_chatGPT.test',
        # 'reddit_cohere.test',
        # 'reddit_davinci.test',
        # 'reddit_dolly.test',
        # 'reddit_flant5.test',
        'wikipedia_chatgpt.test',
        # 'wikipedia_cohere.test',
        # 'wikipedia_davinci.test',
        # 'wikipedia_dolly.test',
    ]
    for file in files:
        with open(base_dir + file, 'r', encoding='utf-8') as row_in_f:
            row_json = json.load(row_in_f)
            human_jsons = [x for x in row_json if x['label'] == 0][0: nums]
        with open(dir + '/' + file + '.' + 'dp' + '.jsonl', 'r', encoding='utf-8') as in_f:
            ai_jsons = []
            for line in in_f:
                ai_jsons.append(json.loads(line))
            ai_jsons = ai_jsons[0: nums]
        results = []
        for human_json in human_jsons:
            results.append({
                'label': 0,
                'content': human_json['content']
            })
        for ai_json in ai_jsons:
            if ai_json['ai_rewrite'] is None:
                continue
            results.append({
                'label': 1,
                'content': ai_json['ai_rewrite']
            })
        with open(dir + '/' + file + '.' + dir + '.test', 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(results))


if __name__ == "__main__":
    # dp = DipperParaphraser()

    # prompt = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:"
    # input_text = "I cannot definitively answer whether your chest pain is related to the intake of Clindamycin and Oxycodone without conducting a thorough examination or reviewing your medical history. However, I can share some information that may help you better understand the potential risks associated with these medications.\n\nClindamycin is an antibiotic that is sometimes associated with gastrointestinal (GI) side effects, including abdominal pain, diarrhea, and nausea. In rare cases, Clindamycin can cause serious GI conditions such as Clostridium difficile-associated diarrhea (CDAD). While chest pain is not a common side effect, it is possible that your GI symptoms could be causing referred pain in your chest.\n\nOxycodone is an opioid pain medication that is sometimes associated with side effects such as respiratory depression, dizziness, and constipation. Rarely, opioids can cause heart-related side effects such as arrhythmias or chest pain.\n\nIt is important to note that chest pain can have many different causes, including heart conditions, lung conditions, and gastroesophageal reflux disease (GERD). Therefore, it is crucial that you contact your healthcare provider as soon as possible to report your symptoms. They may recommend further testing to evaluate the underlying cause of your chest pain.\n\nIn the meantime, you can take the following steps to help manage your symptoms:\n\n1. Continue taking your medications as prescribed, but do not increase the dosage without speaking to your healthcare provider.\n2. Avoid taking large or frequent doses of opioids, as this can increase the risk of side effects.\n3. Stay hydrated by drinking plenty of water or other clear fluids.\n4. Eat smaller, more frequent meals throughout the day instead of large meals.\n5. Avoid caffeine, alcohol, and other substances that can irritate your GI tract.\n6. Practice deep breathing exercises to help reduce anxiety and improve oxygenation to your body.\n\nI hope this information is helpful. I encourage you to contact your healthcare provider with any concerns or questions you may have."

    # print(f"Input = {prompt} <sent> {input_text} </sent>\n")
    # output_l60_sample = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
    # print(f"Output (Lexical diversity = 60, Sample p = 0.75) = {output_l60_sample}\n")

    # print(paraphase(dp, input_text))

    # for file in os.listdir('./dp'):
    #     if file.find('.dp.test') == -1:
    #         continue
    #     print(file)
    #     with open('./dp/' + file, 'r', encoding='utf-8') as in_f:
    #         print(len(json.load(in_f)))

    # generate_datas(dp, './dp_100/', '.dp')
    #
    mix_rewrite_and_human_data('dp_100')

    pass
