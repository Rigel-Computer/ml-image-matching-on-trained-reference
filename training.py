import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet18

# ResNet18 laden, nur Feature-Teil (ohne Klassifikation)
from torchvision.models import ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Bild-Transformation für ResNet18
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Bildpfad → 512-dimensionale Feature-Vektor
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(x).squeeze().numpy()
    return vec


# Alle Bilder eines Verzeichnisses verarbeiten
def load_images(folder):
    features, filenames = [], []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, f)
            try:
                vec = extract_features(path)
                features.append(vec)
                filenames.append(f)
            except Exception as e:
                print(f"Überspringe {f}: {e}")
    return np.array(features), np.array(filenames)


# Vektoren + Dateinamen speichern
if __name__ == "__main__":
    features, names = load_images("images")
    np.save("features.npy", features)
    np.save("filenames.npy", names)
    print(f"{len(features)} Bilder verarbeitet und gespeichert.")
