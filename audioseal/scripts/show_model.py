import torch
from audioseal import AudioSeal

# 選擇裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 AudioSeal 的 generator
model_name = "audioseal_wm_16bits"
generator = AudioSeal.load_generator(model_name).to(device)

# 列印模型架構
print("===== Model Architecture =====")
print(generator)

# 列印模型參數名稱與形狀
print("\n===== Model Parameters =====")
for name, param in generator.named_parameters():
    print(f"{name}: {tuple(param.shape)}")

# 如果想要看某些權重的內容 (例如前幾個數值)
print("\n===== First layer weights (first 5 values) =====")
for name, param in generator.named_parameters():
    print(f"{name} -> {param.view(-1)[:5]}")
    break  # 只看第一層就好