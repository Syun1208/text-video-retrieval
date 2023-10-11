import os
import glob
import sys
from cleanvision import Imagelab
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.abspath(ROOT))
WORK_DIR = os.path.dirname(ROOT)


# directory_path = '/media/hoangtv/New Volume/backup/video/data/Keyframes_L28/'
directory_path = os.path.join(WORK_DIR, 'data/news_aic2023/Keyframes_L28/')



def remove_images_not_in_keep_list(all_images, keep_images):
    for image_path in all_images.copy():  # Tạo một bản sao của danh sách để tránh lỗi khi loại bỏ phần tử
        if image_path not in keep_images:
            try:
                os.remove(image_path)
                print(f"Deleted image: {image_path}")
                all_images.remove(image_path)  # Loại bỏ khỏi danh sách tất cả các ảnh
            except OSError as e:
                print(f"Error deleting image {image_path}: {e}")

def list_subdirectories(path):
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirectories


subdirectories = list_subdirectories(directory_path)

for subdir in subdirectories:
    path = os.path.join(directory_path, subdir)
    print(path)
    # Specify path to folder containing the image files in your dataset
    imagelab = Imagelab(data_path=path)
    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()
    # Produce a neat report of the issues found in your dataset
    imagelab.report()
    image_clean = imagelab.issues.query("is_near_duplicates_issue")
    check = image_clean.index.tolist()[::2]
    remove_images_not_in_keep_list(image_clean.index.tolist(), check)

    imagelab = Imagelab(data_path=path)
    # Automatically check for a predefined list of issues within your dataset
    imagelab.find_issues()
    # Produce a neat report of the issues found in your dataset
    imagelab.report()
    image_clean = imagelab.issues.query("is_near_duplicates_issue")
    check = image_clean.index.tolist()[::2]
    remove_images_not_in_keep_list(image_clean.index.tolist(), check)