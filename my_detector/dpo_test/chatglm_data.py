import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()


def call_with_messages(message):
    response, history = model.chat(tokenizer, message, history=[])
    return response


def generate_paraphase_data(file_path, file_name, output_dir= './glm/'):
    prompt_template = "Please rewrite the following AI-generated text to make it more like human text, {without any useless content}: "
    print(file_path)
    try:
        with open(output_dir + file_name + '.glm.jsonl', 'r', encoding='utf-8') as out_f:
            existed_lines = 0
            for line in out_f:
                existed_lines += 1
            print(existed_lines)
    except Exception as e:
        existed_lines = 0
    with open(file_path, 'r',encoding='utf-8') as in_f:
        cur = 0
        with open(output_dir + file_name + '.glm.jsonl', 'a', encoding='utf-8') as out_f:

            file_objs = json.load(in_f)
            ai_objs = [x for x in file_objs if x['label'] == 1][0:1000]
            results = []

            for ai_obj in tqdm(ai_objs):
                cur+=1
                if cur < existed_lines:
                    continue
                ai_content = ai_obj['content']
                ai_rewrite = call_with_messages(prompt_template + ai_obj['content'])
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
        'cheat_generation.test',
        'cheat_polish.test',
        'ghostbuster_claude.test',
        'hc3_plus_qa_row.test',
        # 'open_qa.academic.test',
        # 'open_qa.continue.test',
        # 'open_qa.difficult.test',
        # 'open_qa.easy.test',
        # 'open_qa.test',
        # 'open_qa.rewrite.test',
        'reddit_chatGPT.test',
        'reddit_cohere.test',
        'reddit_davinci.test',
        'reddit_dolly.test',
        'reddit_flant5.test',
        'wikipedia_chatgpt.test',
        'wikipedia_cohere.test',
        'wikipedia_davinci.test',
        'wikipedia_dolly.test',
    ]
    for file in files:
        with open(base_dir + file, 'r', encoding='utf-8') as row_in_f:
            row_json = json.load(row_in_f)
            human_jsons = [x for x in row_json if x['label'] == 0][0: nums]
        with open(dir + '/' + file + '.' + dir + '.jsonl', 'r', encoding='utf-8') as in_f:
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
            # try:
            #     re.split(r'\s+|\n|\r|\t', ai_json['ai_rewrite'])
            # except Exception as e:
            #     print(ai_json)
            results.append({
                'label': 1,
                'content': ai_json['ai_rewrite']
            })
        with open(dir + '/' + file + '.' + dir + '.test', 'w', encoding='utf-8') as out_f:
            out_f.write(json.dumps(results))

if __name__ == '__main__':
    # print(chat("Please rewrite the following AI-generated text to make it more like human text, {without any useless content}:  I cannot definitively answer whether your chest pain is related to the intake of Clindamycin and Oxycodone without conducting a thorough examination or reviewing your medical history. However, I can share some information that may help you better understand the potential risks associated with these medications.\n\nClindamycin is an antibiotic that is sometimes associated with gastrointestinal (GI) side effects, including abdominal pain, diarrhea, and nausea. In rare cases, Clindamycin can cause serious GI conditions such as Clostridium difficile-associated diarrhea (CDAD). While chest pain is not a common side effect, it is possible that your GI symptoms could be causing referred pain in your chest.\n\nOxycodone is an opioid pain medication that is sometimes associated with side effects such as respiratory depression, dizziness, and constipation. Rarely, opioids can cause heart-related side effects such as arrhythmias or chest pain.\n\nIt is important to note that chest pain can have many different causes, including heart conditions, lung conditions, and gastroesophageal reflux disease (GERD). Therefore, it is crucial that you contact your healthcare provider as soon as possible to report your symptoms. They may recommend further testing to evaluate the underlying cause of your chest pain.\n\nIn the meantime, you can take the following steps to help manage your symptoms:\n\n1. Continue taking your medications as prescribed, but do not increase the dosage without speaking to your healthcare provider.\n2. Avoid taking large or frequent doses of opioids, as this can increase the risk of side effects.\n3. Stay hydrated by drinking plenty of water or other clear fluids.\n4. Eat smaller, more frequent meals throughout the day instead of large meals.\n5. Avoid caffeine, alcohol, and other substances that can irritate your GI tract.\n6. Practice deep breathing exercises to help reduce anxiety and improve oxygenation to your body.\n\nI hope this information is helpful. I encourage you to contact your healthcare provider with any concerns or questions you may have.", 'chatglm3-6b'))
    base_dir = '../../my_detector/roberta_test/data/'
    files = [
        'cheat_generation.test',
        'cheat_polish.test',
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
        'hc3_row.test'
        # 'wikipedia_cohere.test',
        # 'wikipedia_davinci.test',
        # 'wikipedia_dolly.test',
    ]
    mix_rewrite_and_human_data('glm', nums=1000)