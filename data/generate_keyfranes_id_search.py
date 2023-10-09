import os
import json
from pathlib import Path
import sys

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
def main():
    keyframes_id_save = open(os.path.join(ROOT, 'dicts/keyframes_id_search.json'), 'w')
    folder_keyframes = '/media/hoangtv/New Volume/backup/data'
    list_image_paths = []
    for keyframes in sorted(os.listdir(folder_keyframes)):
        if 'Keyframes' not in keyframes:
            break
        print(keyframes)
        if not os.path.isfile(os.path.join(folder_keyframes, keyframes)):
            sorted_LOs = sorted(os.listdir(os.path.join(folder_keyframes, keyframes)))
            for LO in sorted_LOs:
                if os.path.isdir(os.path.join(folder_keyframes,keyframes, LO)):
                    sorted_image_paths = sorted(os.listdir(os.path.join(folder_keyframes,keyframes, LO)))
                    for image_path in sorted_image_paths:
                        if os.path.splitext(image_path)[1] == '.jpg':
                            list_image_paths.append(os.path.join(folder_keyframes, keyframes, LO, image_path).replace('/'.join(folder_keyframes.split("/")[:-1]) + '/', ''))

    json.dump(list_image_paths, keyframes_id_save, indent=6)
    keyframes_id_save.close()

if __name__ == '__main__':
    main()


