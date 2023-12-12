import os

# 用于处理 ghostbuster 那奇怪的数据结构，抽成按行的json文件

# 填入数据根目录
ghostbuster_root_dir = 'D:\\毕设\\数据\\ghostbuster-master\\data'

# 获取某个目录下所有txt文件
def get_all_txt_files(directory):
    txt_files = []
    for file in os.listdir(directory):
        if os.path.splitext(file)[1] == '.txt':
            txt_files.append(file)
    return txt_files


if __name__ == '__main__':
    for domain in os.listdir(ghostbuster_root_dir):
        domain_dir = ghostbuster_root_dir + '\\' + domain
        for model_name in os.listdir(domain_dir):
            if model_name.find('prompt') != -1:
                continue
            else:
                domain_model_dir = domain_dir + '\\' + model_name
                with open('./test_data/ghostbuster/' + domain + '_' + model_name + '.txt', 'w', encoding='utf-8') as out_put:
                    txt_files = get_all_txt_files(domain_model_dir)
                    for txt_file in txt_files:
                        with open(domain_model_dir + '\\' + txt_file, 'r', encoding='utf-8') as f:
                            out_put.write(f.read().replace('\n', '') + '\n')

