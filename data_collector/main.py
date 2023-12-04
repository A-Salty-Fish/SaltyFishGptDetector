# encoding=utf-8

import json
import os

import fitz

import pandas as pd
import requests
import PyPDF2
import textract


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
            print(f'download success:{pdf_id}')
        else:
            print(f'download fail:{pdf_id}')


def convert_pdf_to_txt(pdf_id):
    try:
        reader = fitz.open(f'./row_data/arxiv_pdf/{pdf_id}.pdf')
        text = ""
        for page in reader:
            # text += str(page.get_text("blocks"))
            for block in page.get_text("blocks"):
                text += str(block[4] + '\n')
        with open(f'./row_data/arxiv_text/{pdf_id}.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"text success:{pdf_id}")
    except Exception as e:
        print(e)
        print(f"text fail:{pdf_id}")


if __name__ == '__main__':
    # get_categories()
    # download_pdf('0704.0002')
    # convert_pdf_to_txt('0704.0002')
    # with open('./row_data/categories.txt', 'r') as f:
    #     cn = 0
    #     nn = 0
    #     for line in f:
    #         c = line.split('\t')[0]
    #         n = line.split('\t')[1]
    #         cn += 1
    #         nn += int(n)
    #     print(cn)
    #     print(nn)
    for dirpath, dirnames, filenames in os.walk('D:\\毕设\\数据\\M4-main\\data'):
        for filename in filenames:
            if filename.startswith('README'):
                continue
            with open('D:\\毕设\\数据\\M4-main\\data\\' + filename, 'r', encoding='utf-8') as f:
                i = 0
                for line in f:
                    i += 1
            pre_name = filename.split('.')[0]
            model_name = pre_name.split('_')[-1]
            domain_name = "_".join(pre_name.split('_')[0: -1])
            print(domain_name + '\t' + model_name + '\t' + str(i))
