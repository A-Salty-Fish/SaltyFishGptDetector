import json

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def load_hc3_jsonl(file_name):
    jsons = []
    with open(file_name, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            json_obj = json.loads(line)
            jsons.append({
                'label': 0,
                'content': json_obj['human']
            })
            jsons.append({
                'label': 1,
                'content': json_obj['ai']
            })
    return jsons


def convert_jsons_knn(json_objs, kn=3):
    # 拿出文本
    texts = [x['content'] for x in json_objs]

    # 文本特征提取
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # 聚类
    kmeans = KMeans(n_clusters=kn)
    kmeans.fit(X)

    # 聚类结果
    knn_labels = kmeans.labels_

    knn_json_map = {}
    for i in range(0, kn):
        knn_json_map[str(i)] = []

    for i in range(0, len(texts)):
        json_obj = json_objs[i]
        knn_label = knn_labels[i]
        knn_json_map[str(knn_label)].append(json_obj)

    for i in knn_json_map:
        print(f'{str(i)} : {str(len(knn_json_map[i]))}')

    return knn_json_map


if __name__ == '__main__':
    all_jsons = []
    for file_name in ['finance', 'medicine', 'open_qa', 'wiki_csai']:
        jsons = load_hc3_jsonl('./test_data/hc3_english_mix_multi/' + file_name + '.mix.jsonl')
        all_jsons += jsons

    for kn in [3, 5, 7, 9, 11, 13, 15]:
        knn_map = convert_jsons_knn(all_jsons, kn)
        for knn_label in knn_map:
            with open('./test_data/hc3_english_mix_knn/' + str(kn) + '_' + str(knn_label) + '.jsonl', 'w', encoding='utf-8') as out_f:
                for json_obj in knn_map[knn_label]:
                    out_f.write(json.dumps(json_obj) + '\n')
