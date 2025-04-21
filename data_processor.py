import numpy as np
import pickle
import os

from torch.onnx.symbolic_opset9 import contiguous


def preprocess_H(H):
    """
    將 H 轉換為神經網絡可用的實數格式：
    將實部和虛部分別作為兩個通道。

    原始 H 形狀: [batch, rx, tx, carriers]
    轉換後形狀: [batch, 2, rx, tx, carriers]
    其中第2個維度的通道0是實部，通道1是虛部
    """
    H_real = np.real(H)  # 取得實部
    H_imag = np.imag(H)  # 取得虛部

    # 將實部虛部直接放在第二個維度（axis=1）
    H_combined = np.stack([H_real, H_imag], axis=-4)
    return H_combined



def normalize_H_logarithmic(H, global_H_max):
    """
    使用對數正規化，將 H 的幅值壓縮到範圍 [0, 1]。
    
    Args:
    - H: 複數數據的矩陣，或已經轉換為實數（含實部虛部通道）的矩陣。
    - global_H_max: 全域最大值，用於正規化縮放。

    Returns:
    - H_normalized: 正規化後的矩陣數據。
    """
    if global_H_max > 0:
        if np.iscomplexobj(H):
            # 如果輸入是複數，先計算幅值然後對數正規化
            H_normalized = np.log(1 + np.abs(H)) / np.log(1 + global_H_max)
            return H_normalized
        else:
            # 如果輸入已是實數（具有實部虛部通道的格式）
            # 注意：H 形狀為 [batch, rx, tx, carriers, 2]
            # 計算每個複數的幅值
            H_real = H[:,0]
            H_imag = H[:,1]
            H_magnitude = np.sqrt(H_real ** 2 + H_imag ** 2)

            # 對幅值進行對數正規化
            H_magnitude_normalized = np.log(1 + H_magnitude) / np.log(1 + global_H_max)

            # 保持相位不變，調整幅值
            # 計算原始幅值（避免除以零）
            original_magnitude = np.maximum(np.sqrt(H_real ** 2 + H_imag ** 2), 1e-12)

            # 計算單位向量（保持相位）
            unit_real = H_real / original_magnitude
            unit_imag = H_imag / original_magnitude

            # 應用新的幅值
            normalized_real = unit_real * H_magnitude_normalized
            normalized_imag = unit_imag * H_magnitude_normalized

            # 合併為最終輸出
            return np.stack([normalized_real, normalized_imag], axis=-4)
    else:
        raise ValueError("全域最大值 global_H_max 必須大於 0")


def process_and_save_H(D="train"):
    """
    加載 DeepMIMO 數據，處理 H 矩陣，並保存至本地文件系統。
    - 訓練階段 (D=="train") 時計算並儲存 global_H_max。
    - 測試階段 (D=="test") 時加載已保存的 global_H_max。
    """
    # 確認文件存在
    if not os.path.exists(PICKLE_LOAD_PATH):
        print(f"❌ 找不到 DeepMIMO 數據集 `{PICKLE_LOAD_PATH}`，請先執行 `main_generate.py` 生成數據！")
        return

    # 加載 DeepMIMO 數據
    print(f"📥 正在加載 DeepMIMO 數據：{PICKLE_LOAD_PATH}...")
    with open(PICKLE_LOAD_PATH, "rb") as file:
        dataset = pickle.load(file)

    # 提取 LOS 使用者的數據
    LoS_status = dataset[0]['user']['LoS']
    valid_users = np.where(LoS_status == 1)[0]  # 只選擇 LOS 用戶

    if len(valid_users) == 0:
        print("❌ 沒有有效的 LOS 使用者，請檢查數據集！")
        return

    print(f"✅ 找到 {len(valid_users)} 名有效 LOS 使用者，開始處理 H 矩陣...")

    # Step 1: 提取所有有效使用者的原始複數 H 矩陣
    H_complex = np.array([dataset[0]['user']['channel'][idx] for idx in valid_users])

    if D == "train":
        # 計算 global_H_max 並儲存
        global_H_max = np.max(np.abs(H_complex))
        np.save(GLOBAL_H_MAX_SAVE_PATH, global_H_max)
        print(f"🌟 計算並保存全域最大值 global_H_max: {global_H_max}")
    elif D == "test":
        # 載入 global_H_max
        if os.path.exists(GLOBAL_H_MAX_SAVE_PATH):
            global_H_max = np.load(GLOBAL_H_MAX_SAVE_PATH)
            print(f"📥 載入全域最大值 global_H_max: {global_H_max}")
        else:
            raise FileNotFoundError(f"❌ 找不到已保存的 global_H_max (`{GLOBAL_H_MAX_SAVE_PATH}`)。"
                                    f" 請先處理訓練數據並計算 global_H_max！")
    else:
        raise ValueError("❌ 無效的參數 D，必須為 'train' 或 'test'。")

    # Step 3: 先將 H 轉換成分離實部虛部的格式
    H_processed = np.array([preprocess_H(H) for H in H_complex])

    # Step 4: 對 H_processed 進行對數正規化
    H_all_normalized = normalize_H_logarithmic(H_processed, global_H_max)

    # Step 5: 保存正規化後的 H
    np.save(NORM_H_SAVE_PATH, H_all_normalized)
    print(f"✅ 處理後的 H 矩陣已儲存至 `{NORM_H_SAVE_PATH}`，其形狀為: {H_all_normalized.shape}")

    # 顯示部分處理後的數據進行驗證
    print(f"📊 示例處理數據（第一個樣本的尺寸）: {H_all_normalized[0].shape}")
    print(f"📊 示例數據 - 實部通道前幾個值: {H_all_normalized[0,0].flatten()[:5]}")
    print(f"📊 示例數據 - 虛部通道前幾個值: {H_all_normalized[0,1].flatten()[:5]}")



if __name__ == "__main__":
    # 設定存檔路徑
    PICKLE_LOAD_PATH = "DeepMIMO_trainset.pkl"  # 讀取 DeepMIMO 生成的數據
    NORM_H_SAVE_PATH = "processed_H_train.npy"  # 儲存處理後的 H
    GLOBAL_H_MAX_SAVE_PATH = "global_H_max.npy"  # 儲存全域最大幅值

    process_and_save_H("train")
    H_normalized = np.load(NORM_H_SAVE_PATH)


    # 顯示統計信息 - 注意現在我們有兩個通道
    print(f"H_normalized實部範圍: min={np.min(H_normalized[:,0])}, max={np.max(H_normalized[:,0])}")
    print(f"H_normalized虛部範圍: min={np.min(H_normalized[:,1])}, max={np.max(H_normalized[:,1])}")

    # 計算幅值以檢查正規化效果
    H_magnitude = np.sqrt(H_normalized[:,0] ** 2 + H_normalized[:,1] ** 2)
    print(f"H_normalized幅值範圍: min={np.min(H_magnitude)}, max={np.max(H_magnitude)}")

