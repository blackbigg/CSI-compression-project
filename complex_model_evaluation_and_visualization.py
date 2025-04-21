import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  # for 3D scatter plot


# 評估模型績效(直接使用複數計算)
def evaluate_model(original_path, reconstructed_path, output_dir="results"):
    # 創建結果存儲資料夾
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    originals = np.load(original_path)  # 加載原始數據
    predictions = np.load(reconstructed_path)  # 加載重建數據

    # 確保數據是複數，如果不是則報錯
    if not (np.iscomplexobj(originals) and np.iscomplexobj(predictions)):
        raise ValueError("輸入的 npy 檔案不是複數，請確認數據格式")

    epsilon = 1e-8

    # 複數整體進行誤差計算
    mse = np.mean(np.abs(originals.flatten() - predictions.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(originals.flatten() - predictions.flatten()))
    mape = np.mean(
        np.abs((originals.flatten() - predictions.flatten()) / (np.abs(originals.flatten()) + epsilon))) * 100

    power_total = np.sum(np.abs(originals) ** 2, axis=(1, 2, 3))
    mse_total = np.sum(np.abs(originals - predictions) ** 2, axis=(1, 2, 3))

    valid_mask = power_total > epsilon
    nmse = np.mean(mse_total[valid_mask] / (power_total[valid_mask] + epsilon))

    r2_val = r2_score(np.abs(originals.flatten()), np.abs(predictions.flatten()))

    # NMSE in dB
    nmse_db = 10 * np.log10(nmse + epsilon)

    # === ✅ 加入 rho 指標 ===
    originals_flat = np.abs(originals.flatten())
    predictions_flat = np.abs(predictions.flatten())

    vx = originals_flat - np.mean(originals_flat)
    vy = predictions_flat - np.mean(predictions_flat)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))

    print("\n📊 複數模型評估 (直接使用複數計算):")
    print(f"✅ 均方誤差 (MSE): {mse:.6f}")
    print(f"✅ 均方根誤差 (RMSE): {rmse:.6f}")
    print(f"✅ 平均絕對誤差 (MAE): {mae:.6f}")
    print(f"✅ 平均絕對百分比誤差 (MAPE): {mape:.2f}%")
    print(f"✅ 歸一化均方誤差 (NMSE): {nmse:.6f}")
    print(f"✅ 歸一化均方誤差 (NMSE, dB): {nmse_db:.2f} dB")
    print(f"✅ 決定係數 (R²): {r2_val:.6f}")
    print(f"✅ 皮爾森相關係數 (ρ): {rho:.6f}")

    # 將評估指標存為文字檔
    metrics_file = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_file, "w", encoding="utf-8") as file:
        file.write("複數模型評估指標：\n")
        file.write(f"MSE: {mse:.6f}\nRMSE: {rmse:.6f}\n")
        file.write(f"MAE: {mae:.6f}\nMAPE: {mape:.2f}%\n")
        file.write(f"NMSE: {nmse:.6f}\nNMSE (dB): {nmse_db:.2f} dB\n")
        file.write(f"R²: {r2_val:.6f}\nρ: {rho:.6f}\n")

    print(f"\n📁 評估指標已保存到 '{metrics_file}'")

    visualize_results(originals, predictions, output_dir)


# 視覺化信號成果 (直接使用複數)
def visualize_results(originals, predictions, output_dir):
    plt.figure(figsize=(18, 16))

    originals_real, originals_imag = originals.real, originals.imag
    predictions_real, predictions_imag = predictions.real, predictions.imag

    # 散點圖（預測 vs 實際，以振幅呈現）
    plt.subplot(3, 2, 1)
    plt.scatter(np.abs(originals.flatten()), np.abs(predictions.flatten()), alpha=0.05)
    plt.plot([np.abs(originals).min(), np.abs(originals).max()],
             [np.abs(originals).min(), np.abs(originals).max()], 'r--')
    plt.xlabel("Original Values (Magnitude)")
    plt.ylabel("Predicted Values (Magnitude)")
    plt.title("Original vs Predicted (Magnitude)")

    # 誤差分佈直方圖 (以振幅差異)
    plt.subplot(3, 2, 2)
    errors = np.abs(originals.flatten()) - np.abs(predictions.flatten())
    plt.hist(errors, bins=50, color='g', alpha=0.6)
    plt.xlabel("Error (Magnitude)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution Histogram (Magnitude)")

    # 第一個樣本的熱圖展示
    index = 0  # 樣本索引
    depth = 0  # 深度索引

    plt.subplot(3, 2, 3)
    sns.heatmap(originals_real[index, depth, :, :], cmap="viridis")
    plt.title("Original Data (Real Part)")

    plt.subplot(3, 2, 4)
    sns.heatmap(originals_imag[index, depth, :, :], cmap="viridis")
    plt.title("Original Data (Imaginary Part)")

    plt.subplot(3, 2, 5)
    sns.heatmap(predictions_real[index, depth, :, :], cmap="viridis")
    plt.title("Predicted Data (Real Part)")

    plt.subplot(3, 2, 6)
    sns.heatmap(predictions_imag[index, depth, :, :], cmap="viridis")
    plt.title("Predicted Data (Imaginary Part)")

    plt.tight_layout()
    results_path = os.path.join(output_dir, "results_visualization.png")
    plt.savefig(results_path)
    plt.show()
    print(f"\n📁 視覺化圖表已保存到 '{results_path}'")

    # 3D 圖部分（存檔）
    originals_real_data = originals_real[index, :, :, :]
    predictions_real_data = predictions_real[index, :, :, :]
    errors_real_data = np.abs(originals_real_data - predictions_real_data)  # 使用絕對誤差

    rx_vals, tx_vals, carriers_vals = np.meshgrid(
        np.arange(originals_real_data.shape[0]),  # rx
        np.arange(originals_real_data.shape[1]),  # tx
        np.arange(originals_real_data.shape[2])  # carriers
    )

    rx_vals_flat = rx_vals.flatten()
    tx_vals_flat = tx_vals.flatten()
    carriers_vals_flat = carriers_vals.flatten()
    originals_flat = originals_real_data.flatten()
    predictions_flat = predictions_real_data.flatten()
    errors_flat = errors_real_data.flatten()

    # 原始數據 3D 圖
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter(rx_vals_flat, tx_vals_flat, carriers_vals_flat,
                     c=originals_flat, cmap='viridis', s=10, alpha=0.8)
    fig.colorbar(sc1, ax=ax, shrink=0.5, pad=0.05)
    ax.set_xlabel("Rx")
    ax.set_ylabel("Tx")
    ax.set_zlabel("Carriers")
    ax.set_title("3D Scatter Plot: Original Data (Real Part)")
    plt.savefig(os.path.join(output_dir, "3D_Original_Data.png"))
    plt.show()

    # 預測數據 3D 圖
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc2 = ax.scatter(rx_vals_flat, tx_vals_flat, carriers_vals_flat,
                     c=predictions_flat, cmap='viridis', s=10, alpha=0.8)
    fig.colorbar(sc2, ax=ax, shrink=0.5, pad=0.05)
    ax.set_xlabel("Rx")
    ax.set_ylabel("Tx")
    ax.set_zlabel("Carriers")
    ax.set_title("3D Scatter Plot: Predicted Data (Real Part)")
    plt.savefig(os.path.join(output_dir, "3D_Predicted_Data.png"))
    plt.show()

    # 絕對誤差 3D 圖
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc3 = ax.scatter(rx_vals_flat, tx_vals_flat, carriers_vals_flat,
                     c=errors_flat, cmap='cividis', s=10, alpha=0.8)
    fig.colorbar(sc3, ax=ax, shrink=0.5, pad=0.05)
    ax.set_xlabel("Rx")
    ax.set_ylabel("Tx")
    ax.set_zlabel("Carriers")
    ax.set_title("3D Scatter Plot: Absolute Error |Original - Predicted|")
    plt.savefig(os.path.join(output_dir, "3D_Absolute_Error.png"))
    plt.show()

    print(f"\n🌟 All files have been saved to '{output_dir}'!")


# 主程式執行區塊
if __name__ == "__main__":
    original_path = "output_csi_original.npy"
    reconstructed_path = "output_csi_reconstructed.npy"
    evaluate_model(original_path, reconstructed_path)
