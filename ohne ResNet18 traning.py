import os
import cv2
import numpy as np

def extract_features(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    return img.flatten() / 255.0

def load_images(folder):
    features, filenames = [], []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, f)
            features.append(extract_features(path))
            filenames.append(f)
    return np.array(features), np.array(filenames)

if __name__ == "__main__":
    features, names = load_images("images")
    np.save("features.npy", features)
    np.save("filenames.npy", names)
    print(f"{len(features)} Bilder gespeichert.")
