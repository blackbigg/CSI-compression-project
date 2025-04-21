import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # CUDA deterministic 設定，使 GPU 訓練結果可穩定重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset
class MIMODataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 複數操作
def custom_abs(data):
    real, imag = data[:, 0], data[:, 1]
    return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # 加入epsilon防止sqrt(0)


def custom_angle(data):
    real, imag = data[:, 0], data[:, 1]
    return torch.atan2(imag, real)


def to_complex(data):
    return torch.complex(data[:, 0], data[:, 1])


# Loss Functions (加入MSE Loss)

def spectral_loss(y_true, y_pred):
    y_true_abs, y_pred_abs = custom_abs(y_true), custom_abs(y_pred)
    abs_loss = F.mse_loss(y_pred_abs, y_true_abs)

    y_true_angle, y_pred_angle = custom_angle(y_true), custom_angle(y_pred)
    angle_diff = (y_true_angle - y_pred_angle + torch.pi) % (2 * torch.pi) - torch.pi
    angle_loss = torch.mean(angle_diff ** 2)

    return abs_loss + 0.5* angle_loss + 1e-8


def nmse_loss(y_true, y_pred):
    y_true_complex, y_pred_complex = to_complex(y_true), to_complex(y_pred)
    error_power = torch.mean(torch.abs(y_true_complex - y_pred_complex) ** 2)
    signal_power = torch.mean(torch.abs(y_true_complex) ** 2) + 1e-8
    return error_power / signal_power


def mse_loss(y_true, y_pred):
    y_true_complex, y_pred_complex = to_complex(y_true), to_complex(y_pred)
    return torch.mean(torch.abs(y_true_complex - y_pred_complex) ** 2)


# 模型定義（改進版）
class SEAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEAttention, self).__init__()

        reduced_channels = max(channel // reduction, 1)  # 保證不會變成0

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)



class ResidualBlock(nn.Module):
    def __init__(self, img_channels=2):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv3d(img_channels, 8, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(8, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv3d(8, 16, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(16, eps=1e-03, momentum=0.99),
                                  nn.LeakyReLU(negative_slope=0.3),
                                  nn.Conv3d(16, img_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(img_channels, eps=1e-03, momentum=0.99))

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3)

    def forward(self, x):
        ori_x = x

        # concatenate
        x = self.conv(x) + ori_x

        return self.leakyRelu(x)


class MIMOAutoEncoder(nn.Module):
    def __init__(self, encoding_dim=32, dropout_rate=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm3d(16),
            nn.Dropout3d(dropout_rate),
            SEAttention(channel=16),  # <-- 加入注意力模塊


            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm3d(32),
            nn.Dropout3d(dropout_rate),
            SEAttention(channel=32),  # <-- 加入注意力模塊

            nn.Conv3d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01), nn.BatchNorm3d(2),
            nn.Dropout3d(dropout_rate),
            SEAttention(channel=2),  # <-- 加入注意力模塊



            nn.Flatten(),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * 32 * 2 * 32, encoding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 2 * 32 * 2 * 32),
            nn.Unflatten(1, (2, 32, 2, 32)),



            nn.ConvTranspose3d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.01),
            ResidualBlock(32),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.01),
            ResidualBlock(16),

            nn.ConvTranspose3d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(0.01),
            ResidualBlock(2),

            nn.Conv3d(2, 2, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




# 調整版的訓練函式：支援不同 loss function 組合
def train_model_curriculum(model, train_loader, val_loader, device, stage, epochs, patience):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    counter = 0  # early stopping 計數器

    train_losses = []
    val_losses = []

    print(f"🔔 開始訓練階段 {stage} ...")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)

            # 根據階段切換 loss function 組合
            if stage == 1:
                loss = mse_loss(batch, output)
            elif stage == 2:
                loss = mse_loss(batch, output) + nmse_loss(batch, output)
            elif stage == 3:
                loss = mse_loss(batch, output) + nmse_loss(batch, output) + spectral_loss(batch, output)
            else:
                raise ValueError("未知的階段！")


            if torch.isnan(loss):
                print(f"Epoch {epoch + 1}: Loss 為 NaN，停止訓練！")
                return train_losses, val_losses

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 驗證階段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                if stage == 1:
                    loss = mse_loss(batch, output)
                elif stage == 2:
                    loss = mse_loss(batch, output) + nmse_loss(batch, output)
                elif stage == 3:
                    loss = mse_loss(batch, output) + nmse_loss(batch, output) + spectral_loss(batch, output)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Stage {stage}] Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), f"saved_models/mimo_autoencoder_stage{stage}.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"✅ Early stopping at epoch {epoch+1} in stage {stage}")
                break
    # ⭐ 加在訓練迴圈之後，確保最佳權重
    model.load_state_dict(torch.load(f"saved_models/mimo_autoencoder_stage{stage}.pth"))
    # 再次儲存確保最終最佳模型安全
    torch.save(model.state_dict(), f"saved_models/best_final_model_stage{stage}.pth")

    return train_losses, val_losses

def visualize_reconstruction(model, dataset, device, num_samples=5):
    model.eval()
    plt.figure(figsize=(18, 6))

    for i in range(num_samples):
        # 加入batch維度並移至裝置
        original = torch.tensor(dataset[i]).unsqueeze(0).to(device)

        # 模型重建資料
        with torch.no_grad():
            reconstructed = model(original)

        # 此處將複數轉為單純的幅值（取絕對值）
        original_csi = np.abs(to_complex(original.cpu()).numpy().flatten())
        reconstructed_csi = np.abs(to_complex(reconstructed.cpu()).numpy().flatten())

        # 原始CSI資料繪製
        plt.subplot(2, num_samples, i + 1)
        plt.plot(original_csi, label='Original')
        plt.title(f'Original CSI {i + 1}')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # 重建CSI資料繪製
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.plot(reconstructed_csi, label='Reconstructed', color='r')
        plt.title(f'Reconstructed CSI {i + 1}')
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png')
    plt.show()

# 🌟 主程式改為三階段訓練
def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置：{device}")

    # 建立儲存模型資料夾
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    # 檢查資料完整性
    data_np = np.load("processed_H_train.npy")
    assert not np.isnan(data_np).any(), "資料含有 NaN!"
    assert not np.isinf(data_np).any(), "資料含有 Inf!"
    print('資料最大值:', data_np.max(), '資料最小值:', data_np.min())
    # 讀取 dataset
    dataset = MIMODataset("processed_H_train.npy")
    # **新增 Validation Set**
    train_size = int(0.8 * len(dataset))  # 80% 給 Training
    val_size = len(dataset) - train_size  # 20% 給 Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MIMOAutoEncoder().to(device)
    print(f"模型總參數數量: {count_parameters(model):,}")

    all_train_losses = []
    all_val_losses = []

    for stage in range(1, 4):
        if stage > 1:
            model.load_state_dict(
                torch.load(f"saved_models/mimo_autoencoder_stage{stage - 1}.pth", map_location=device))

        train_losses, val_losses = train_model_curriculum(
            model, train_loader, val_loader, device,
            stage=stage, epochs=50, patience=10
        )

        all_train_losses += train_losses
        all_val_losses += val_losses

    plt.figure()
    plt.plot(all_train_losses, label="Training Loss")
    plt.plot(all_val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss vs Epoch (3-stage Curriculum Learning)")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss_curriculum.png')
    plt.show()

    visualize_reconstruction(model, dataset, device, num_samples=5)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

