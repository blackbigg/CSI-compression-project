import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import savemat
from train_model_3phase import MIMOAutoEncoder, MIMODataset

# 設定參數（請依您目前架構調整）
encoding_dim = 32  # 請依您的具體情形調整
model_path = 'saved_models/best_final_model_stage3.pth'
test_data_path = 'processed_H_test.npy'
global_H_max_path = 'global_H_max.npy'
output_path_mat = 'output_csi.mat'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 對數反正規化函數
def logarithmic_denormalization(H_normalized, global_H_max):
    H_real = H_normalized[:, 0]
    H_imag = H_normalized[:, 1]

    # 正規化後數據的幅值
    H_magnitude_normalized = np.sqrt(H_real ** 2 + H_imag ** 2)

    # 還原至原始數據的幅值 (inverse operation)
    H_magnitude_original = np.power(1 + global_H_max, H_magnitude_normalized) - 1

    # 保留相位資訊
    original_magnitude_norm = np.maximum(H_magnitude_normalized, 1e-12)
    unit_real = H_real / original_magnitude_norm
    unit_imag = H_imag / original_magnitude_norm

    # 計算還原後的實部與虛部資料
    H_real_original = unit_real * H_magnitude_original
    H_imag_original = unit_imag * H_magnitude_original

    # 合併成複數數據
    return H_real_original + 1j * H_imag_original



# 存儲結果至.mat與.npy檔
def save_original_and_reconstructed_to_mat_and_npy(original, reconstructed, output_path):
    mat_dict = {
        'original_data': original,
        'reconstructed_data': reconstructed
    }
    savemat(output_path, mat_dict)
    np.save(output_path.replace('.mat', '_original.npy'), original)
    np.save(output_path.replace('.mat', '_reconstructed.npy'), reconstructed)


# Load 測試資料及global max
test_data = np.load(test_data_path)
global_H_max = np.load(global_H_max_path)

# Tensor格式資料處理
test_dataset = MIMODataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"測試數據形狀: {test_dataset.data.shape}")
print(f"全域最大值 (global_H_max): {global_H_max}")

# 導入模型
model = MIMOAutoEncoder(encoding_dim).to(device)  # 注意這裡是否與原定義相符
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("🚀 開始進行測試...")
# 計算模型輸出，並進行複數合併與反正規化（已加入shape列印）
original_data = []
reconstructed_data = []

with torch.no_grad():
    for data_batch in test_loader:


        data = data_batch.to(device)

        print(f"data shape: {data.shape}")

        recon_data = model(data)

        print(f"recon datashape: {recon_data.shape}")



        original_denorm = logarithmic_denormalization(data.cpu().numpy(), global_H_max)
        recon_denorm = logarithmic_denormalization(recon_data.cpu().numpy(), global_H_max)



        original_data.append(original_denorm)
        reconstructed_data.append(recon_denorm)



# 轉換成numpy格式
original_data = np.concatenate(original_data, axis=0)
reconstructed_data = np.concatenate(reconstructed_data, axis=0)

print(f"✅ 原始數據形狀: {original_data.shape}")
print(f"✅ 重建數據形狀: {reconstructed_data.shape}")

# 將結果存儲為mat和npy檔案
save_original_and_reconstructed_to_mat_and_npy(original_data, reconstructed_data, output_path_mat)

print(f"✅ 原始數據和重建數據已儲存為: {output_path_mat}")

