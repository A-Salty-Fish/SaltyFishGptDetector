import csv
import json
import os

if __name__ == '__main__':
    loss_map = {'rewrite': 0.4930301308631897, 'continue': 0.2966940999031067, 'academic': 0.4215707778930664,
                'difficult': 0.993345320224762, 'easy': 0.3965831696987152, 'qa': 0.29204198718070984}

    import math

    batch_num = 16

    log_loss_map = {}
    log_loss_sum = 0
    log_loss_percent_map = {}

    for k in loss_map:
        log_loss_map[k] = (-math.log10(1 - loss_map[k]))
        log_loss_sum += log_loss_map[k]

    for k in log_loss_map:
        log_loss_percent_map[k] = log_loss_map[k] / log_loss_sum

    sorted_dict = dict(sorted(log_loss_percent_map.items(), key=lambda item: item[1], reverse=True))
    sorted_list = [[k, sorted_dict[k]] for k in sorted_dict]

    adversary_nums_map = {}

    for key in loss_map:
        adversary_nums_map[key] = 1

    for kv in sorted_list:
        adversary_nums_map[kv[0]] += int(0.5 + kv[1] * (batch_num - len(loss_map)))

    print(adversary_nums_map)

    # import pandas as pd
    # knn_path = 'D:\\毕设\\数据\\实验结果\\KNN'
    # all_files = os.listdir(knn_path)
    #
    # print(all_files)
    #
    # file_3 = [x for x in all_files if x.find('_3') != -1]
    # file_5 = [x for x in all_files if x.find('_5') != -1]
    # file_7 = [x for x in all_files if x.find('_7') != -1]
    # file_9 = [x for x in all_files if x.find('_9') != -1]
    # file_11 = [x for x in all_files if x.find('_11') != -1]
    #
    # print(file_3)
    # print(file_5)
    # print(file_7)
    # print(file_9)
    # print(file_11)
    # # Specify the CSV files to be merged
    # # csv_files = ['file1.csv', 'file2.csv', 'file3.csv']
    # #
    # # # Read each CSV file into a DataFrame
    #
    # dfs_3 = [pd.read_csv(knn_path + '\\' + file) for file in file_3]
    # dfs_5 = [pd.read_csv(knn_path + '\\' + file) for file in file_5]
    # dfs_7 = [pd.read_csv(knn_path + '\\' + file) for file in file_7]
    # dfs_9 = [pd.read_csv(knn_path + '\\' + file) for file in file_9]
    # dfs_11 = [pd.read_csv(knn_path + '\\' + file) for file in file_11]
    # #
    # # # Concatenate the DataFrames
    # # df_merged = pd.concat(dfs)
    # #
    # # # Save the merged DataFrame to a new CSV file
    # # df_merged.to_csv('merged_file.csv', index=False)
    #
    # pd.concat(dfs_3).to_csv(knn_path + '\\' + 'merged_3.csv', index=False)
    # pd.concat(dfs_5).to_csv(knn_path + '\\' + 'merged_5.csv', index=False)
    # pd.concat(dfs_7).to_csv(knn_path + '\\' + 'merged_7.csv', index=False)
    # pd.concat(dfs_9).to_csv(knn_path + '\\' + 'merged_9.csv', index=False)
    # pd.concat(dfs_11).to_csv(knn_path + '\\' + 'merged_11.csv', index=False)

    # department_map = {
    #
    # }
    # with open('./json_array', 'r', encoding='utf-8') as f:
    #     json_obj = json.load(f)
    #     data = json_obj['data']
    #     for d in data:
    #         department = d['department']
    #         if department_map.get(d['department']) is None:
    #             department_map[department] = []
    #         department_map[department].append(d)
    #
    # for k in department_map:
    #     if k is None:
    #         continue
    #     with open(k + '.csv', 'w') as f:
    #         if (len(department_map[k])) == 0:
    #             continue
    #         fieldnames = department_map[k][0].keys()
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         # 写入标题行
    #         writer.writeheader()
    #
    #         # 写入数据行
    #         writer.writerows(department_map[k])
