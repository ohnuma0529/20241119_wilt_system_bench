import cv2
import sys
from ultralytics import YOLO

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

#bboxどうしのIoUを計算する関数
def calc_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

def get_best_bbox(bbox_list, df, frame_number, key_list):
    for i in key_list:
        bbox = [df[str(i) + "_bbox_x1"].dropna().iloc[-1], df[str(i) + "_bbox_y1"].dropna().iloc[-1], df[str(i) + "_bbox_x2"].dropna().iloc[-1], df[str(i) + "_bbox_y2"].dropna().iloc[-1]]
        best_iou = 0
        for bbox2 in bbox_list:
            iou = calc_iou(bbox, bbox2)
            if iou > best_iou:
                best_iou = iou
                best_bbox = bbox2
        if best_iou < 0.25:
            best_bbox = bbox
        # dfを更新
        df.loc[frame_number, str(i) + '_bbox_x1'] = best_bbox[0]
        df.loc[frame_number, str(i) + '_bbox_y1'] = best_bbox[1]
        df.loc[frame_number, str(i) + '_bbox_x2'] = best_bbox[2]
        df.loc[frame_number, str(i) + '_bbox_y2'] = best_bbox[3]
        df.loc[frame_number, str(i) + '_center_x'] = (best_bbox[0] + best_bbox[2]) / 2
        df.loc[frame_number, str(i) + '_center_y'] = (best_bbox[1] + best_bbox[3]) / 2
    return df

# 更新されていないbboxがある場合、そのbboxを削除する関数
def check_bbox(df, key_list):
    if len(df[str(key_list[0]) + '_bbox_x1'].dropna()) < 10:
        return df, key_list
    for i in key_list:
        if len(df[str(i) + '_bbox_x1'].dropna()) < 10:
            continue
        # 時点前のbbox
        bbox10 = [df[str(i) + "_bbox_x1"].dropna().iloc[-10], df[str(i) + "_bbox_y1"].dropna().iloc[-10], df[str(i) + "_bbox_x2"].dropna().iloc[-10], df[str(i) + "_bbox_y2"].dropna().iloc[-10]]
        bbox5 = [df[str(i) + "_bbox_x1"].dropna().iloc[-5], df[str(i) + "_bbox_y1"].dropna().iloc[-5], df[str(i) + "_bbox_x2"].dropna().iloc[-5], df[str(i) + "_bbox_y2"].dropna().iloc[-5]]
        bbox3 = [df[str(i) + "_bbox_x1"].dropna().iloc[-3], df[str(i) + "_bbox_y1"].dropna().iloc[-3], df[str(i) + "_bbox_x2"].dropna().iloc[-3], df[str(i) + "_bbox_y2"].dropna().iloc[-3]]
        bbox = [df[str(i) + "_bbox_x1"].dropna().iloc[-1], df[str(i) + "_bbox_y1"].dropna().iloc[-1], df[str(i) + "_bbox_x2"].dropna().iloc[-1], df[str(i) + "_bbox_y2"].dropna().iloc[-1]]
        # 4時点のbboxの座標が同じなら削除
        if bbox == bbox3 == bbox5 == bbox10:
            remove_list = [col for col in df.columns if col.startswith(str(i) + "_")]
            df.drop(columns=remove_list, inplace=True)
            # key_listから排除
            key_list.remove(i)
    return df, key_list
# IOUが0.8以上のbboxがある場合、数字の小さいほうのbboxを削除する関数
def check_bbox2(df, key_list):
    for i in key_list:
        bbox = [df[str(i) + "_bbox_x1"].dropna().iloc[-1], df[str(i) + "_bbox_y1"].dropna().iloc[-1], df[str(i) + "_bbox_x2"].dropna().iloc[-1], df[str(i) + "_bbox_y2"].dropna().iloc[-1]]
        for j in key_list:
            if i == j:
                continue
            bbox2 = [df[str(j) + "_bbox_x1"].dropna().iloc[-1], df[str(j) + "_bbox_y1"].dropna().iloc[-1], df[str(j) + "_bbox_x2"].dropna().iloc[-1], df[str(j) + "_bbox_y2"].dropna().iloc[-1]]
            iou = calc_iou(bbox, bbox2)
            if iou > 0.8:
                if i > j:
                    remove_list = [col for col in df.columns if col.startswith(str(i) + "_")]
                    df.drop(columns=remove_list, inplace=True)
                    key_list.remove(i)
                else:
                    remove_list = [col for col in df.columns if col.startswith(str(j) + "_")]
                    df.drop(columns=remove_list, inplace=True)
                    key_list.remove(j)
    return df, key_list

#10分前の角度と現在の角度の差を計算
# 全葉の角度変化の中央値の倍を閾値として、その値を超える葉を削除する関数
# 角度はラジアン
def check_bbox3(df, key_list):
    angle_diff_list = []
    for i in key_list:
        if len(df[str(i) + '_angle'].dropna()) < 35:
            return df, key_list
        # angle_diffの絶対値を計算してリストに追加
        angle_diff = abs(df[str(i) + "_angle"].dropna().iloc[-1] - df[str(i) + "_angle"].dropna().iloc[-30])
        angle_diff_list.append(angle_diff)
    # 絶対値の差の中央値に基づく閾値を設定
    if len(df[str(key_list[0]) + '_angle'].dropna()) > 30:
        threshold = 1
    else:
        threshold = np.median(angle_diff_list) * 20
    # print("threshold:", threshold)
    # print("angle_diff_list:", angle_diff_list)

    for i, angle_diff in zip(key_list, angle_diff_list):
        if angle_diff > threshold:  # 既に絶対値をとっているので条件はそのまま
            remove_list = [col for col in df.columns if col.startswith(str(i) + "_")]
            df.drop(columns=remove_list, inplace=True)
            key_list.remove(i)
    return df, key_list

#bboxリストと深度推定画像より、bbox毎の平均深度を計算する関数
#depth0-255で深度を表現する二次元配列
# bboxリストと深度推定画像より、bbox毎の平均深度を計算し、深度でソートする関数
def calc_depth(bbox_list, depth):
    depth_info = []

    for bbox in bbox_list:
        x1, y1, x2, y2 = map(int, bbox)
        # 各bbox領域の平均深度を計算し、bbox座標と深度のペアを追加
        avg_depth = np.mean(depth[y1:y2, x1:x2])
        depth_info.append((bbox, avg_depth))

    # 深度でソートして返す
    depth_info.sort(key=lambda x: x[1], reverse=True)

    # bboxリストと深度リストをそれぞれ分離して返す
    sorted_bbox_list = [item[0] for item in depth_info]
    sorted_depth_list = [item[1] for item in depth_info]

    return [sorted_bbox_list, sorted_depth_list]


# bboxの縁が画像の縁に近い場合、bboxを削除する関数
#bbox_listはbbox_list[0]にbboxの座標、bbox_list[1]に深度が格納されている
#bbox削除時は深度も一緒に削除する
def remove_edge_bbox(bbox_list, height, width, rate=0.95):
    center_x = width // 2
    center_y = height // 2
    # 矩形のサイズを計算
    rect_width = int(width * rate)
    rect_height = int(height * (rate - 0.1))

    #左上
    start_x = center_x - rect_width // 2
    start_y = center_y - rect_height // 2
    #右下
    end_x = center_x + rect_width // 2
    # end_y = center_y
    end_y = center_y + rect_height // 2
    
    #bboxの線が縁に含まれていたらbbox_listから削除
    new_bbox_list = []
    new_depth_list = []
    for i, bbox in enumerate(bbox_list[0]):
        x1, y1, x2, y2 = bbox
        if x1 > start_x and y1 > start_y and x2 < end_x and y2 < end_y:
            #座標と深度をnew_bbox_listに追加
            # new_bbox_list.append(bbox_list[:, i])    
            new_bbox_list.append(bbox)
            new_depth_list.append(bbox_list[1][i])
        
    # return new_bbox_list
    return [new_bbox_list, new_depth_list]

def get_first_bbox(bbox_list, depth, height=1024, width=1024, leaf_num=5):
    con_list = calc_depth(bbox_list, depth)
    con_list = remove_edge_bbox(con_list, height, width)
    #con_list[0]のbbox座標から、bboxの面積を計算し、画像サイズの半分以上だったら削除
    new_bbox_list = []
    new_depth_list = []
    for i, bbox in enumerate(con_list[0]):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area <= (height * width) / 2:
            new_bbox_list.append(bbox)
            new_depth_list.append(con_list[1][i])
            if len(new_bbox_list) == leaf_num:
                break
    new_list = [new_bbox_list, new_depth_list]
    # print(new_list)
    #上位5つのbboxを返す
    return new_list

def get_frame(frame, bbox):
    x1, y1, x2, y2 = bbox
    # print("input:", x1, y1, x2, y2)
    #bboxの大きさを1.3倍に拡大
    x1 = int(x1 - (x2 - x1) * 0.15)
    x2 = int(x2 + (x2 - x1) * 0.15)
    y1 = int(y1 - (y2 - y1) * 0.15)
    y2 = int(y2 + (y2 - y1) * 0.15)
    if x1 < 0:
        x1 = 0
    elif x2 > frame.shape[1]:
        x2 = frame.shape[1]
    if y1 < 0:
        y1 = 0
    elif y2 > frame.shape[0]:
        y2 = frame.shape[0]
    # print("output:", x1,y1,x2,y2)
    return frame[y1:y2, x1:x2], x1, y1, x2, y2

def set_df(df, con_list, key_list, frame_number):
    # 初期化
    columns = df.columns.tolist() if len(df.columns) > 0 else []

    if len(df.columns) == 0:
        # データフレームが空の場合、1から始めてカラムを定義
        for i in range(len(con_list[0])):
            columns.append(str(i+1) + '_bbox_x1')
            columns.append(str(i+1) + '_bbox_y1')
            columns.append(str(i+1) + '_bbox_x2')
            columns.append(str(i+1) + '_bbox_y2')
            columns.append(str(i+1) + '_base_x')
            columns.append(str(i+1) + '_base_y')
            columns.append(str(i+1) + '_tip_x')
            columns.append(str(i+1) + '_tip_y')
            columns.append(str(i+1) + '_length')
            columns.append(str(i+1) + '_center_x')
            columns.append(str(i+1) + '_center_y')
            key_list.append(i+1)
        df = pd.DataFrame(columns=columns)
        return df, key_list, 0
    else:
        # データフレームに既存のカラムがある場合、続きの番号から追加
        last_num = int(columns[-1].split('_')[0])
        for i in range(last_num+1, last_num+1+len(con_list[0])):
            df.loc[frame_number, str(i) + '_bbox_x1'] = 0
            df.loc[frame_number, str(i) + '_bbox_y1'] = 0
            df.loc[frame_number, str(i) + '_bbox_x2'] = 0
            df.loc[frame_number, str(i) + '_bbox_y2'] = 0
            df.loc[frame_number, str(i) + '_base_x'] = 0
            df.loc[frame_number, str(i) + '_base_y'] = 0
            df.loc[frame_number, str(i) + '_tip_x'] = 0
            df.loc[frame_number, str(i) + '_tip_y'] = 0
            df.loc[frame_number, str(i) + '_length'] = 0
            df.loc[frame_number, str(i) + '_center_x'] = 0
            df.loc[frame_number, str(i) + '_center_y'] = 0
            df.loc[frame_number, str(i) + '_angle'] = 0
            key_list.append(i)
        #既存のdfに新しいカラムを追加
        # df = df.reindex(columns=columns)
        print("new_key_list:", key_list)
        return df, key_list, last_num

# 萎れ計算用関数
#dfとleaf_numを受け取り、その番号の葉の萎れを計算してdfに追加
def cal_wilt(df, leaf_num, detect_size):
    # df["str(leaf_num)" + "_bbox_x1"]が存在しない場合はエラーを返す
    if not str(leaf_num) + "_bbox_x1" in df.columns:
        print(leaf_num)
        raise ValueError("There is no such column in the DataFrame.")
    if len(df) == 1:
        df[str(leaf_num) + "_wilt"] = 0
        return df
    #最も古いデータから最新のデータまでの平均を計算
    #1. 葉の長さの平均を計算
    leaf_length = df.loc[:, str(leaf_num) + '_length'].mean()
    # 初期値が一定範囲内なら，初期値-最新値をとる
    if np.pi / 2 <= df[str(leaf_num) + "_angle"].dropna().iloc[0] <= 3 * np.pi / 2:
        df[str(leaf_num) + "_angle_diff"] = df[str(leaf_num) + "_angle"].dropna().iloc[0] - df[str(leaf_num) + "_angle"]
    else:
        df[str(leaf_num) + "_angle_diff"] = df[str(leaf_num) + "_angle"] - df[str(leaf_num) + "_angle"].dropna().iloc[0]
    # angleを計算時は左上原点にしているため、sinを-1倍する
    df[str(leaf_num) + "_sin"] = np.sin(df[str(leaf_num) + "_angle_diff"]) * -1
    #3. Bbox中心Y座標の変化を計算
      # 左上原点ではなく左下原点にしてから計算
    df[str(leaf_num) + "_center_diff"] = (detect_size - df[str(leaf_num) + "_center_y"]) - (detect_size - df[str(leaf_num) + "_center_y"].dropna().iloc[0])
    df[str(leaf_num) + "_std_center_diff"] = df[str(leaf_num) + "_center_diff"] / leaf_length
    #4. 萎れ指標(sin2 + 3/1)を計算して格納
    # df[str(leaf_num) + "_wilt"] = -np.sin(df[str(leaf_num) + "_angle_diff"]) + (df[str(leaf_num) + "_center_diff"]) / leaf_length
    df[str(leaf_num) + "_wilt"] = df[str(leaf_num) + "_sin"] + df[str(leaf_num) + "_std_center_diff"]
    # 平滑化処理
    df[str(leaf_num) + "_wilt"] = df[str(leaf_num) + "_wilt"].rolling(5).mean().interpolate()
    return df
def vis_df(df, leaf_num, column_name, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[str(leaf_num) + column_name], label=column_name)
    plt.xlabel("frame")
    plt.ylabel(column_name)
    plt.title("leaf" + str(leaf_num) + column_name)
    plt.legend()
    plt.savefig(save_dir + "/" + str(leaf_num) + column_name + ".png")
    plt.close()

# 全てのwiltを用いて最終的なwiltを計算する関数
# wiltの値の平均を計算
# nan以外の数値の行が30行未満のwiltは使わない
def calc_final_wilt(df, key_list, frame_number, lifetime=30):
    wilt_list = []
    sin_list = []
    center_diff_list = []
    for i in key_list:
        if len(df[str(i) + "_wilt"].dropna()) < lifetime:
            continue
        # 最新のwiltを用いて計算し、格納
        wilt = df[str(i) + "_wilt"].dropna().iloc[-1]
        wilt_list.append(wilt)
        # sin
        sin = df[str(i) + "_sin"].dropna().iloc[-1]
        sin_list.append(sin)
        # center_diff
        center_diff = df[str(i) + "_std_center_diff"].dropna().iloc[-1]
        center_diff_list.append(center_diff)
    # 平均を計算
    final_wilt = np.mean(wilt_list)
    final_sin = np.mean(sin_list)
    final_center_diff = np.mean(center_diff_list)
    # dfに格納
    df.loc[frame_number, "final_wilt"] = final_wilt
    df.loc[frame_number, "final_sin"] = final_sin
    df.loc[frame_number, "final_center_diff"] = final_center_diff
    return df

def vis_wilt(df, leaf_number, save_dir):
    num = leaf_number
    # leaf6_sin, leaf6_std_center_diff, leaf6_wiltをまとめて可視化
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[f"{num}_sin"].rolling(5).mean().interpolate(), label="sin")
    plt.plot(df.index, df[f"{num}_std_center_diff"].rolling(5).mean().interpolate(), label="std_center_diff")
    plt.plot(df.index, df[f"{num}_wilt"].rolling(5).mean().interpolate(), label="wilt")
    plt.xlabel("frame")
    plt.ylabel("value")
    plt.title(f"leaf{num}")  # 日本語フォント指定
    plt.legend()
    # plt.show()
    plt.savefig(f"{save_dir}/leaf{num}_wilts.png")

def convert_num(num):
    if num < 10:
        return f"0{num}"
    else:
        return str(num)

def vis_past_wilt(area, column_name, save_dir=None):
    str_area = convert_num(area)
    # データを読み込む
    df = pd.read_csv(f"output/20241119/{str_area}_20181127/{str_area}_20181127.csv", index_col=0)
    df_irr = pd.read_csv("df_irrigation.csv", index_col=0)
    df_irr = df_irr[f"{str(area)}_irrigation"]
    wilt_df = pd.read_csv("df_all_wilt.csv", index_col=0)
    # 01で正規化
    wilt_df = (wilt_df - wilt_df.min()) / (wilt_df.max() - wilt_df.min())

    plt.figure(figsize=(10, 5))

    # 各葉のデータをプロット
    if column_name == "final_wilt" or column_name == "final_sin" or column_name == "final_center_diff":
        plt.plot(
            df.index,
            df[column_name].interpolate(),
            label="final_wilt",
            color="red"
        )
    else:
        for i in range(1, 100):
            try:
                plt.plot(
                    df.index,
                    df[f"{i}_{column_name}"].rolling(5).mean().interpolate(),
                    label=f"leaf{i}_{column_name}"
                )
            except KeyError:
                continue

    # 灌漑データを縦の点線としてプロット
    for idx, val in enumerate(df_irr):
        if val == 1:  # 灌漑が行われたフレームのみ
            plt.axvline(x=idx, color="magenta", linestyle="--", alpha=0.7, label="irrigation" if idx == 0 else "")

    plt.plot(
        wilt_df.index,
        wilt_df[f"{area}_wilt"],
        label="past_wilt",
        color="black"
    )

    # ラベルと凡例
    plt.xlabel("frame")
    plt.ylabel(f"{column_name}")
    plt.title(f"area{area}_{column_name}")
    plt.legend()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/area{area}_{column_name}.png")
    else:
        plt.show()

def vis_stem(area, column_name, save_dir=None):
    str_area = convert_num(area)
    # データを読み込む
    df = pd.read_csv(f"output/20241119/{str_area}_20181127/{str_area}_20181127.csv", index_col=0)
    df_irr = pd.read_csv("df_irrigation.csv", index_col=0)
    df_irr = df_irr[f"{area}_irrigation"]
    df_stem = pd.read_csv("df_irrigation.csv", index_col=0)
    df_stem = df_stem[f"{area}_stem"]

    df_past_wilt = pd.read_csv("df_all_wilt.csv", index_col=0)

    #df[column_name].interpolate()を正規化
    new_wilt = (df[column_name].interpolate() - df[column_name].interpolate().min()) / (df[column_name].interpolate().max() - df[column_name].interpolate().min())
    past_wilt = (df_past_wilt[f"{area}_wilt"] - df_past_wilt[f"{area}_wilt"].min()) / (df_past_wilt[f"{area}_wilt"].max() - df_past_wilt[f"{area}_wilt"].min())

    plt.figure(figsize=(10, 5))

    # 各葉のデータをプロット
    if column_name == "final_wilt" or column_name == "final_sin" or column_name == "final_center_diff":
        plt.plot(
            df.index,
            new_wilt,
            label="new_wilt",
            color="red"
        )
    else:
        for i in range(1, 100):
            try:
                plt.plot(
                    df.index,
                    df[f"{i}_{column_name}"].rolling(5).mean().interpolate(),
                    label=f"leaf{i}_{column_name}"
                )
            except KeyError:
                continue

    # 灌漑データを縦の点線としてプロット
    for idx, val in enumerate(df_irr):
        if val == 1:  # 灌漑が行われたフレームのみ
            plt.axvline(x=idx, color="magenta", linestyle="--", alpha=0.7, label="irrigation" if idx == 0 else "")

    plt.plot(
        df_stem.index,
        df_stem,
        label="stem",
        color="black"
    )

    plt.plot(
        df_past_wilt.index,
        past_wilt,
        label="past_wilt",
        color="blue"
    )

    # ラベルと凡例
    plt.xlabel("frame")
    plt.ylabel(f"{column_name}")
    plt.title(f"area{area}_{column_name}")
    plt.legend()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/area{area}_{column_name}.png")
    else:
        plt.show()
    # 0~500の範囲で，new_wiltとstem、past_wiltとstem、new_wiltとpast_wiltの相関係数を計算
    # new_wiltの最初のindexを取得
    new_wilt_start = new_wilt.first_valid_index()
    # new_wilt_startから500を利用
    new_wilt = new_wilt.loc[new_wilt_start:new_wilt_start+500]
    new_wilt_size = len(new_wilt)
    stem = df_stem.loc[new_wilt_start:new_wilt_start+new_wilt_size-1]
    past_wilt = past_wilt.loc[new_wilt_start:new_wilt_start+new_wilt_size-1]
    corr1 = np.corrcoef(new_wilt, stem)[0, 1]
    corr2 = np.corrcoef(past_wilt, stem)[0, 1]
    corr3 = np.corrcoef(new_wilt, past_wilt)[0, 1]
    print(f"new_wilt and stem: {corr1}")
    print(f"past_wilt and stem: {corr2}")
    print(f"new_wilt and past_wilt: {corr3}")


def video_process(dataset_path, output_dir, leaf_num, start_frame=0, end_frame=500, server_flg=True):
    # dataset_path = dataset/imgsz_512/01_20181127
    save_name = dataset_path.split("/")[-2] + "_" + dataset_path.split("/")[-1]
    images = Path(dataset_path).glob("*.jpg")
    # 01_20181127_0.jpgの最後の数字順にソート 0, 1, 2, ...
    images = sorted(images, key=lambda x: int(str(x).split("_")[-1].replace(".jpg", "")))
    if server_flg:
        detect = YOLO("weights/detect/20241106_detect.pt")
        pose = YOLO("weights/pose/syn_aug_800_4.pt")
    else:
        detect = YOLO('light_weights/detect/20241106_detect_saved_model/20241106_detect_float32.tflite')
        pose = YOLO('light_weights/pose/syn_800_2_saved_model/syn_800_2_float32.tflite')

    frame_number = 0
    save_dir = '{}/{}'.format(output_dir,save_name)
    os.makedirs(save_dir, exist_ok=True)

    # leaf_num = 30
    pose_size = 640
    detect_size = 1024
    # start_frame = 0
    # end_frame = 500

    #再検出フラグ
    re_detect = False
    re_detect_cnt = 0

    key_list = []

    encoder = "vits"  # モデルの種類を指定
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(torch.load(f"Depth-Anything-V2/depth_anything_v2_{encoder}.pth", map_location="cpu"))
    depth_model.eval()
    depth_model = depth_model.to('cuda')

    # DataFrameの初期化
    df = pd.DataFrame()
    df.index.name = 'frame'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    # 動画のフレームを処理
    for image in images:
        try:
            #計測用start_time計測
            start_time = time.time()
            with open('/sys/class/thermal/thermal_zone0/temp') as t: #CPU温度読み込み
                initial_temperature = int(t.read()) / 1000
            frame = cv2.imread(str(image))
            if frame_number > end_frame:
                break
            if frame_number < start_frame:
                frame_number += 1
                continue
            print("-----------------", frame_number, "-----------------\n")
            frame = cv2.resize(frame, (1024, 1024))
            #初期bbox決定
            if frame_number==start_frame or re_detect:
                height, width, _ = frame.shape
                depth = depth_model.infer_image(frame)

                depth_image = depth_model.infer_image(frame)
                depth = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255  # 0-255にスケーリング
                depth = depth.astype(np.uint8)
                results = detect.predict(frame, imgsz=1024, conf=0.4)
                for result in results:
                    bbox_list = [box.xyxy[0].tolist() for box in result.boxes]
                    con_list = get_first_bbox(bbox_list, depth, height, width, leaf_num=leaf_num)
                    df, key_list, last_num = set_df(df, con_list, key_list, frame_number)
                    for i in range(len(con_list[0])):
                            x1, y1, x2, y2 = con_list[0][i]
                            clip, new_x1, new_y1, new_x2, new_y2 = get_frame(frame, [x1, y1, x2, y2])
                            clip = cv2.resize(clip, (pose_size, pose_size))
                            #pose推定
                            pose_results = pose.predict(clip, imgsz=640, conf=0.1)
                            xys = pose_results[0].keypoints.xy[0].tolist()
                            #clip画像中の座標を取得
                            base_x, base_y = xys[0][0], xys[0][1]
                            tip_x, tip_y = xys[1][0], xys[1][1]
                            #clip画像中の座標をframe画像中の座標に変換
                            base_x = base_x /pose_size * (new_x2 - new_x1) + new_x1
                            base_y = base_y /pose_size * (new_y2 - new_y1) + new_y1
                            tip_x = tip_x /pose_size * (new_x2 - new_x1) + new_x1
                            tip_y = tip_y /pose_size * (new_y2 - new_y1) + new_y1
                            
                            # dfに保存
                            df.loc[frame_number, str(i+1+last_num) + '_bbox_x1'] = x1
                            df.loc[frame_number, str(i+1+last_num) + '_bbox_y1'] = y1
                            df.loc[frame_number, str(i+1+last_num) + '_bbox_x2'] = x2
                            df.loc[frame_number, str(i+1+last_num) + '_bbox_y2'] = y2
                            df.loc[frame_number, str(i+1+last_num) + '_base_x'] = base_x
                            df.loc[frame_number, str(i+1+last_num) + '_base_y'] = base_y
                            df.loc[frame_number, str(i+1+last_num) + '_tip_x'] = tip_x
                            df.loc[frame_number, str(i+1+last_num) + '_tip_y'] = tip_y
                            df.loc[frame_number, str(i+1+last_num) + '_length'] = np.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
                            df.loc[frame_number, str(i+1+last_num) + '_center_x'] = (x1 + x2) / 2
                            df.loc[frame_number, str(i+1+last_num) + '_center_y'] = (y1 + y2) / 2
                            #角度（ラジアン）保存
                            df.loc[frame_number, str(i+1+last_num) + '_angle'] = np.arctan2(tip_y - base_y, tip_x - base_x)

                            df = cal_wilt(df, i+1+last_num, detect_size)
                df = calc_final_wilt(df, key_list, frame_number)
                re_detect = False
                re_detect_cnt = 0
            else:
                results = detect.predict(frame, imgsz=1024, conf=0.4)
                #prev_bbox_listと今回のbbox_listを比較し、それぞれのbboxに対して最もIoUが高いbboxを今回のbboxとする
                for result in results:
                    bbox_list = [box.xyxy[0].tolist() for box in result.boxes]
                    df = get_best_bbox(bbox_list, df, frame_number, key_list)
                    df, key_list = check_bbox(df, key_list)
                    df, key_list = check_bbox2(df, key_list)
                    for i in key_list:
                        # print("key", i)
                        x1 = df.loc[frame_number, str(i) + '_bbox_x1']
                        y1 = df.loc[frame_number, str(i) + '_bbox_y1']
                        x2 = df.loc[frame_number, str(i) + '_bbox_x2']
                        y2 = df.loc[frame_number, str(i) + '_bbox_y2']
                        clip, new_x1, new_y1, new_x2, new_y2 = get_frame(frame, [x1, y1, x2, y2])
                        clip = cv2.resize(clip, (pose_size, pose_size))
                        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                        #pose推定
                        pose_results = pose.predict(clip, imgsz=640, conf=0.1)
                        xys = pose_results[0].keypoints.xy[0].tolist()
                        #clip画像中の座標を取得
                        base_x, base_y = xys[0][0], xys[0][1]
                        tip_x, tip_y = xys[1][0], xys[1][1]
                        #clip画像中の座標をframe画像中の座標に変換
                        base_x = base_x /pose_size * (new_x2 - new_x1) + new_x1
                        base_y = base_y /pose_size * (new_y2 - new_y1) + new_y1
                        tip_x = tip_x /pose_size * (new_x2 - new_x1) + new_x1
                        tip_y = tip_y /pose_size * (new_y2 - new_y1) + new_y1

                        # dfに保存
                        df.loc[frame_number, str(i) + '_base_x'] = base_x
                        df.loc[frame_number, str(i) + '_base_y'] = base_y
                        df.loc[frame_number, str(i) + '_tip_x'] = tip_x
                        df.loc[frame_number, str(i) + '_tip_y'] = tip_y
                        df.loc[frame_number, str(i) + '_length'] = np.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
                        df.loc[frame_number, str(i) + '_angle'] = np.arctan2(tip_y - base_y, tip_x - base_x)
                        df, key_list = check_bbox3(df, key_list)
                        #現在のiがkey_listに含まれていなければcontinue
                        df = cal_wilt(df, i, detect_size)
                if (len(key_list) < 3 and re_detect == False) or (re_detect_cnt > 180 and re_detect == False):
                    re_detect = True
                else:
                    # check_bbox3は大体30時点後に実行されるため
                    if frame_number > start_frame + 15:
                        re_detect_cnt += 1
                df = calc_final_wilt(df, key_list, frame_number)
            #保存
            frame_number += 1
            #計測用end_time計測
            end_time = time.time()
            with open('/sys/class/thermal/thermal_zone0/temp') as t: #CPU温度読み込み
                final_temperature = int(t.read()) / 1000
            #計測結果保存用txt
            with open(os.path.join(save_dir, "result.txt"), "a") as f:
                f.write(f"{frame_number}: {end_time - start_time} sec, {final_temperature} ℃\n")
            #次回計測用に10秒スリープ
            # time.sleep(10)
        except Exception as e:
            print(e)
            frame_number += 1
            continue
    #csv保存
    df.to_csv(os.path.join(save_dir, f"{save_name}.csv"))

img_size_list = [512, 1024, 2048]
source_dir = "dataset"
#imgsize
output_dir = "output_imgsz"
for img_size in img_size_list:
    for area in range(1, 12):
        try:
            dataset_path = f"{source_dir}/imgsz_{img_size}/{convert_num(area)}_20181127"
            video_process(dataset_path, output_dir, 30, start_frame=0, end_frame=30, server_flg=True)
        except Exception as e:
            print(e)
            continue
#重み（1024固定）
img_size = 1024
output_dir = "output_weight"
for area in range(1, 12):
    try:
        dataset_path = f"{source_dir}/imgsz_{img_size}/{convert_num(area)}_20181127"
        video_process(dataset_path, output_dir, 30, start_frame=0, end_frame=30, server_flg=True)
    except Exception as e:
        print(e)
        continue