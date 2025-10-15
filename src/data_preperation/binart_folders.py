import os,shutil
from tqdm import tqdm

SRC_FOLDER="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\processed_jpg"
DEST_FOLDER="C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\processed_binary"


#this is for mapping the 5 claases to two
mapping = {
    "Koilocytotic": "abnormal",
    "Dyskeratotic": "abnormal",
    "Metaplastic": "abnormal",
    "Parabasal": "normal",
    "Superficial-Intermediate": "normal"
}
os.makedirs(DEST_FOLDER, exist_ok=True)


for name,binary_class in mapping.items():
    src_dir=os.path.join(SRC_FOLDER,name)
    dest_dir = os.path.join(DEST_FOLDER, binary_class)
    os.makedirs(dest_dir,exist_ok=True)

    for f in tqdm(os.listdir(src_dir), desc=f"{name} → {binary_class}"):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dest_dir, f"{name}_{f}")   #avoid duplicate file ames
        shutil.copy(src_path, dst_path)

