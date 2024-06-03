import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
"""
every label: [869, 213, 16, 3]
the number of file I selected: 2231
"""


def find_all_data():
    root_path = "F:/DataBase/Voice/svd1"
    neg_dir, pos_dir = "healthy", "pathological"
    healthy_files = glob.glob("{root}/{dir}/*/*/*/*.wav".format(root=root_path, dir=neg_dir))
    symptom_files = glob.glob("{root}/{dir}/*/*/*/*.wav".format(root=root_path, dir=pos_dir))
    print(len(healthy_files))
    print(len(symptom_files))
    return healthy_files, [0] * len(healthy_files), symptom_files, [1] * len(symptom_files)


def get_all_label():
    json_dict = None
    root_path = "F:/DataBase/Voice/svd1"
    with open(root_path + "/data.json", 'r', encoding='utf_8') as fp:
        json_dict = json.load(fp)
    print(type(json_dict))
    name2label = {}
    cnts = [0, 0, 0, 0]
    lines = []
    for item in json_dict:
        # print(item)
        sesss = json_dict[item]["sessions"]
        gender = json_dict[item]["gender"]
        if len(sesss) == 0:
            continue
        for it in sesss:
            bilab = it["classification"]
            sid = it["session_id"]
            if bilab == "healthy":
                # print(bilab)
                name2label[item] = 0
                cnts[0] += 1
                filepath = root_path + f"/healthy/{gender}/{item}/{sid}/"
                if not os.path.exists(filepath):
                    continue
                for fitem in os.listdir(filepath):
                    lines.append((filepath + fitem, 0))
                    break
                continue

            di = it["pathologies"]
            if ',' in di:
                parts = di.split(',')
                for ite in parts:
                    di = ite.strip()
                    print(di)
                    filepath = root_path + f"/pathological/{gender}/{item}/{sid}/"
                    print(filepath)
                    if not os.path.exists(filepath):
                        continue
                    if di == "Hyperfunktionelle Dysphonie":
                        name2label[item] = 1
                        cnts[1] += 1
                        for j, fitem in enumerate(os.listdir(filepath)):
                            lines.append((filepath + fitem, 1))
                            if j > 4:
                                break

                    if di == "Hypofunktionelle Dysphonie":
                        name2label[item] = 2
                        cnts[2] += 1
                        for fitem in os.listdir(filepath):
                            lines.append((filepath + fitem, 2))
                    if di == "GERD":
                        name2label[item] = 3
                        cnts[3] += 1
                        for fitem in os.listdir(filepath):
                            lines.append((filepath + fitem, 3))
            else:
                print(di)
                filepath = root_path + f"/pathological/{gender}/{item}/{sid}/"
                print(filepath)
                if not os.path.exists(filepath):
                    continue
                if di == "Hyperfunktionelle Dysphonie":
                    name2label[item] = 1
                    cnts[1] += 1
                    for j, fitem in enumerate(os.listdir(filepath)):
                        lines.append((filepath + fitem, 1))
                        if j > 4:
                            break
                if di == "Hypofunktionelle Dysphonie":
                    name2label[item] = 2
                    cnts[2] += 1
                    for fitem in os.listdir(filepath):
                        lines.append((filepath + fitem, 2))
                if di == "GERD":
                    name2label[item] = 3
                    cnts[3] += 1
                    for fitem in os.listdir(filepath):
                        lines.append((filepath + fitem, 3))

    print(cnts)
    print(len(lines))
    with open("./dataset.txt", 'w') as fout:
        for line in lines:
            fout.write(line[0] + ',' + str(line[1]) + '\n')


if __name__ == '__main__':
    get_all_label()
