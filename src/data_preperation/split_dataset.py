import os ,shutil,random
from tqdm import tqdm

DATA_DIR="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\processed_binary"
OUT_DIR="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\split"
split=0.8 #80 train 20 test
random.seed(42)

binary_classes=["normal","abnormal"]
os.makedirs(OUT_DIR,exist_ok=True)

for binary_class in binary_classes:
    files=os.listdir(os.path.join(DATA_DIR,binary_class))
    random.shuffle(files)
    spilt_idx=int(split * len(files))


    training_data=files[:spilt_idx]
    testing_data=files[spilt_idx:]

    for subset,subset_files in [("train", training_data), ("test", testing_data)]:
        subset_dir=os.path.join(OUT_DIR,subset,binary_class)
        os.makedirs(subset_dir,exist_ok=True)

        for f in tqdm(subset_files,desc=f"{binary_class}->{subset}"):
            shutil.copy(os.path.join(DATA_DIR,binary_class,f),os.path.join(subset_dir,f))

