import DeepMIMOv3_noPL
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pickle
import os

from DeepMIMOv3_noPL.visualization import plot_LoS_status

# 存檔路徑
PICKLE_SAVE_PATH = "DeepMIMO_testset.pkl"

def load_or_generate_deepmimo_data(parameters, save_path):
    """
    加載或生成 DeepMIMO 數據。
    如果存在保存的 .pkl 檔案，則直接加載。
    否則，生成數據並保存。
    """
    if os.path.exists(save_path):
        print(f"📥 正在加載數據集：{save_path}...")
        with open(save_path, "rb") as file:
            dataset = pickle.load(file)
        print("✅ 數據集加載成功！")
    else:
        print("📥 未找到存檔，正在生成新的 DeepMIMO 數據...")
        dataset = DeepMIMOv3_noPL.generate_data(parameters)
        print("✅ 數據生成成功！")

        # 保存數據到 .pkl 文件
        with open(save_path, "wb") as file:
            pickle.dump(dataset, file)
        print(f"✅ 數據已保存至：{save_path}")

    return dataset


# 設置參數
parameters = DeepMIMOv3_noPL.default_params()
parameters.update({
    'scenario': 'O1_28',  # 設置場景名稱
    'dataset_folder': '../dataset',
    'num_paths': 5,
    'active_BS': np.array([4]),
    'user_rows': np.arange(1500,1700),  # 選擇特定範圍的使用者
    'user_subsampling':1.0,  # 隨機選擇使用者
    'ue_antenna': {
        'shape': np.array([8, 4]),  # 水平4個元素，垂直2個元素
        'spacing': 0.5,  # 天線間距設為0.5倍波長
        'rotation': np.array([0, 0, 0]),
        'FoV': np.array([180, 180]),  # 水平方向視場角為120度，垂直方向為180度
        'radiation_pattern': 'halfwave-dipole'  # 使用半波偶極輻射模式

    },
    'bs_antenna': {
        'shape': np.array([2, 1]),  # 水平4個元素，垂直2個元素
        'spacing': 0.5,  # 天線間距設為0.5倍波長
        'rotation': np.array([0, 0, 180]),
        'FoV': np.array([180, 180]),  # 水平方向視場角為120度，垂直方向為180度
        'radiation_pattern': 'halfwave-dipole'  # 使用半波偶極輻射模式
    },

    'OFDM_channels': 1,  # 啟用 OFDM
    'OFDM': {
        'bandwidth': 0.00384,
        'subcarriers': 128,
        'selected_subcarriers': list(range(32)),
        'RX_filter': 0  # 添加缺失的參數以避免 KeyError
    }

})

# 加載或生成數據
dataset = load_or_generate_deepmimo_data(parameters, PICKLE_SAVE_PATH)

# 計算總使用者數量
total_users = len(dataset[0]['user']['LoS'])
print(f"📊 總使用者數量: {total_users}")

# 獲取 LOS 狀態
LoS_status = dataset[0]['user']['LoS']
valid_users = np.where(LoS_status == 1)[0]  # 只選擇 LOS 狀態的有效使用者

print(f"⚠️ 有效的使用者數量: {len(valid_users)}")

    # 打印每個選中的使用者的 H
print(f"✅ 第 1 名使用者的 H:")
H_sample = dataset[0]['user']['channel'][0]
print(f"使用者索引 {0} 的 H:")
pprint(H_sample)
print("-" * 40)  # 分隔線



for bs_idx in [0]:
    bs_location = dataset[bs_idx]['location']
    LoS_status = dataset[bs_idx]['user']['LoS']
    user_location = dataset[bs_idx]['user']['location']
    plt.figure()
    plot_LoS_status(bs_location, user_location, LoS_status,0.3)

    plt.title(f'BS{bs_idx+1}')

plt.show()


H_all = dataset[0]['user']['channel']  # 取得所有用戶的 H
print(f"H shape: {H_all.shape}")

# 計算 H 的統計信息
H_magnitude = np.abs(H_all)  # 計算複數 H 的絕對值（振幅）
print(f"H 最小值: {H_magnitude.min()}")
print(f"H 最大值: {H_magnitude.max()}")
print(f"H 平均值: {H_magnitude.mean()}")
print(f"H 標準差: {H_magnitude.std()}")


np.save('H_matrix.npy', H_all)
print("✅ H_matrix 已成功保存為 H_matrix.npy")


