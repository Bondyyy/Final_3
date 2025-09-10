import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, dropout_rate=0.3):
    """
    Tạo một mô hình EfficientNetV2-S với lớp classifier tùy chỉnh.
    """
    # Sử dụng weights mới nhất được khuyến nghị
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_model(model, model_path):
    """
    Tải trọng số đã huấn luyện vào mô hình.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model