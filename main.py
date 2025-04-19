import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity

# ResNet18 nur bis zum Feature-Output laden
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Bild → Tensor transformieren
transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Bildpfad → Vektor (512 Werte)
def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(x).squeeze().numpy()
    return vec


# Alle Bilder im Zielordner vergleichen
def scan_and_compare(folder, train_features, train_names, threshold=0.89):
    for dirpath, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if fname.startswith("Match_"):
                continue

            full = os.path.join(dirpath, fname)
            try:
                vec = extract_features(full).reshape(1, -1)
                sims = cosine_similarity(vec, train_features)[0]
                best = np.argmax(sims)
                score = sims[best]
                print(f"{fname} → Score: {score:.3f}")

                if score > threshold:
                    new_name = os.path.join(dirpath, f"Match_{fname}")
                    os.rename(full, new_name)
                    print(f"Match erkannt, umbenannt → Match_{fname}")
            except Exception as e:
                print(f"Fehler bei {fname}: {e}")


if __name__ == "__main__":
    features = np.load("features.npy")
    names = np.load("filenames.npy", allow_pickle=True)

    scan_and_compare("zu_pruefen", features, names, threshold=0.89)
