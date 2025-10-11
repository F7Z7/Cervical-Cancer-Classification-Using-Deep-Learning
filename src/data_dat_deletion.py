import os

ROOT_DIR = "C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data"

for root, folders, files in os.walk(ROOT_DIR):
    deleted_in_folder = 0
    for file in files:
        if file.lower().endswith('.dat'):
            os.remove(os.path.join(root, file))
            deleted_in_folder += 1
    if deleted_in_folder > 0:
        print(f"Deleted {deleted_in_folder} .dat files in folder '{root}'")
 