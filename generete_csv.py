import os
import cv2
import numpy as np
import slideio
import geojson
import pandas as pd
from tqdm import tqdm


def cut_wsi2(wsi_path, geo_dir, save_dir, threhold, b=256, save_loc=False):
    # save location of sub pic rather than sub pic
    name = wsi_path.split('\\')[-1].replace('.', '-')
    d = os.path.join(save_dir, f'{name}.csv')

    slide = slideio.open_slide(wsi_path, "SVS")
    scene = slide.get_scene(0)

    annotation_geojson_path = geo_dir
    w, h = scene.size

    # Load the geojson information
    annotation_geojson_file = open(annotation_geojson_path)
    collection = geojson.load(annotation_geojson_file)
    try:
        if "features" in collection.keys():
            collection = collection["features"]
        elif "geometries" in collection.keys():
            collection = collection["geometries"]
    except AttributeError:
        # already a list?
        pass

    # Read the original image
    image_masked = np.zeros(shape=(int(h / 4), int(w / 4)), dtype=np.float32)
    # image_masked = image_masked.astype(np.float32)

    # Load the coordinates and store them into a numpy array
    for i in range(len(collection)):
        polygon_numpy = np.array(collection[i].geometry.coordinates, dtype=object)
        for j in range(polygon_numpy.shape[0]):
            polygon_numpyd = np.array(polygon_numpy[j])
            polygon_numpy2d = np.array(polygon_numpyd / 4, dtype='int32')
            # 判断新范围是否与大范围重叠
            mask_small = np.zeros_like(image_masked)
            cv2.fillPoly(mask_small, [polygon_numpy2d], color=255)

            image_masked[np.where(np.logical_and(image_masked != 0, mask_small != 0))] = 0
            image_masked[np.where(np.logical_and(image_masked == 0, mask_small != 0))] = 255
    col = ['bag_id', 'x', 'y', 'wsi']
    data = []

    b = int(b / 4)
    m, n = int(h // b / 4), int(w // b / 4)  # m x n  bag
    sel = [0] * m * n

    for i in range(1, m - 1):  # traverse all bag
        for j in range(1, n - 1):
            lu = [i * b * 4, j * b * 4]  # left_up (h,w)
            bag_id = i * n + j

            mask = image_masked[int(i * b): int(i * b + b), int(j * b): int(j * b + b)]
            temp = np.count_nonzero(mask == 255)
            sel[bag_id] = temp / mask.size

            if sel[bag_id] > threhold and save_loc:
                data.append([bag_id, lu[1], lu[0], wsi_path])
                # win = scene.read_block((int(lu[1]), int(lu[0]), b * 4, b * 4),
                #                        (b * 4, b * 4))

    t = pd.DataFrame(columns=col, data=data)
    t.to_csv(d, index=False)


def convert_to_char(value):
    # 将字符串中的数字和空格去掉，只保留后面的部分
    return value.split('\\')[-1].split('.')[0]


def generate_train_test(df):
    # 步骤1：打乱原始DataFrame的顺序
    df = df.sample(frac=1)
    df = df.loc[df['生长方式'].isin(['纤维型', '膨胀型'])]

    # 创建保存'a'和'b'的行，并从中取出20%的DataFrame
    df1 = df.loc[df['生长方式'].isin(['纤维型'])]
    df1_sample = df1.sample(frac=0.2)

    df2 = df.loc[df['生长方式'].isin(['膨胀型'])]
    df2_sample = df2.sample(frac=0.2)

    other_df = pd.concat([df1_sample, df2_sample])
    new_df = df[~df.index.isin(other_df.index)]

    return new_df, other_df


def create_dataset(ori_dir, output_dir):
    # 获取文件夹中所有csv文件的名称
    csv_files = [f for f in os.listdir(ori_dir) if f.endswith('.csv')]

    # 初始化一个空的DataFrame来保存所有数据
    all_data = pd.DataFrame()
    rest_data = pd.read_csv('data/merged_file.csv', encoding='gbk')

    # 遍历所有csv文件，读取内容并添加到all_data中
    for csv_file in csv_files:
        file_path = os.path.join(ori_dir, csv_file)
        data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, data], ignore_index=True)

    all_data['ID'] = all_data['wsi'].apply(convert_to_char)
    merged_df = pd.merge(all_data, rest_data, on='ID', how='outer')

    train_data, test_data = generate_train_test(merged_df)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_data.to_csv(train_path, index=False, encoding='gbk')
    test_data.to_csv(test_path, index=False, encoding='gbk')


if __name__ == '__main__':
    threhold = 0.2  # 一张图中肿瘤区域的占比大小, b=256时修改为0.8
    b = 4096  # if patch size=256, or 4096
    assert b in [256, 4096]

    save_path = os.path.join('data', 'csv_' + str(b))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for name in tqdm(os.listdir('data/SVS_ALL')):
        cut_wsi2(os.path.join('data/SVS_ALL', name), os.path.join('data/geojson_ALL', name.replace('svs', 'geojson')),
                 save_path, threhold=threhold, b=b, save_loc=True)

    create_dataset(save_path, 'data/')