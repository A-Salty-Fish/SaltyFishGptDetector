import csv
import json

if __name__ == '__main__':
    department_map = {

    }
    with open('./json_array', 'r', encoding='utf-8') as f:
        json_obj = json.load(f)
        data = json_obj['data']
        for d in data:
            department = d['department']
            if department_map.get(d['department']) is None:
                department_map[department] = []
            department_map[department].append(d)

    for k in department_map:
        if k is None:
            continue
        with open(k + '.csv', 'w') as f:
            if (len(department_map[k])) == 0:
                continue
            fieldnames = department_map[k][0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 写入标题行
            writer.writeheader()

            # 写入数据行
            writer.writerows(department_map[k])