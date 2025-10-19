if __name__ == "__main__":
    import os
    import torch
    import numpy as np
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm


    TRAIN_DIR = "C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\split\\train"
    TEST_DIR  = "C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\split\\test"
    OUTPUT_DIR = "C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\features"
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    train_set = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_set  = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final FC layer
    model.eval().to(DEVICE)


    def extract_features(dataloader, split_name):
        all_features, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc=f"Extracting {split_name} features"):
                imgs = imgs.to(DEVICE)
                feats = model(imgs)
                feats = feats.view(feats.size(0), -1)  # safer than squeeze
                all_features.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())

            X = np.vstack(all_features)
            y = np.concatenate(all_labels)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f"resnet152_{split_name}_features.npz")
            np.savez_compressed(out_path, X=X, y=y)
            print(f"Saved {split_name} features → {out_path} | X: {X.shape}, y: {y.shape}")

    # --- Run feature extraction ---
    extract_features(train_loader, "train")
    extract_features(test_loader, "test")
