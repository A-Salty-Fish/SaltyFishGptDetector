import json

import pandas as pd
import requests


# 获取目录
def get_categories():
    categories_map = {}
    with open('./row_data/arxiv-metadata-oai-snapshot.json', 'r') as input_file:
        for line in input_file:
            json_obj = json.loads(line)
            k = str(json_obj['categories'])
            if categories_map.get(k) is None:
                categories_map[k] = 1
            else:
                categories_map[k] += 1
    with open('./row_data/categories.txt', 'w') as output_file:
        for k in categories_map:
            output_file.write("" + str(k) + "\t" + str(categories_map[k]) + '\n')


def download_pdf(pdf_id):
    with open(f'./row_data/arxiv_pdf/{pdf_id}.pdf', 'wb') as f:
        r = requests.get(f"https://arxiv.org/pdf/{pdf_id}.pdf")
        if r.status_code == 200:
            f.write(r.content)
            print(f'success:{pdf_id}')
        else:
            print(f'fail:{pdf_id}')


if __name__ == '__main__':
    # get_categories()
    download_pdf('0704.0001')
