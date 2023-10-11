import numpy as np
import json
import tqdm

path_to_save = "/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data"
outfile = f"{path_to_save}/info_ocr_official.npy"

f = open('/home/hoangtv/Desktop/Long/text-video-retrieval/data/dicts/keyframes_id_search.json')
df_tx = open('/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data/infor_search.json')

arr_to_save = np.load("/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data/demo_npy.npy")
data_json = json.load(f)
data_ocr_json = json.load(df_tx)
idx_json = 0
idx_txt = 0
arr_to_save_npy = None
arr_for_empty = np.ones((1,768))*0.00000000001

data_json_keyframes_period =  data_json[0].split("/")[2]
arr_finnal_to_save_npy = None
try:
    for idx_json in tqdm.tqdm(range(len(data_json))):
        data_json_keyframes = data_json[idx_json].split("/")[2]
        # print(data_json[idx_json] ==  data_ocr_json[idx_txt])
        if data_json_keyframes != data_json_keyframes_period:
            if arr_finnal_to_save_npy is None:
                arr_finnal_to_save_npy = arr_to_save_npy
                arr_to_save_npy = None
            else:
                arr_finnal_to_save_npy = np.concatenate((arr_finnal_to_save_npy, arr_to_save_npy), axis = 0)
                arr_to_save_npy = None

        if data_json[idx_json] == data_ocr_json[idx_txt]:
            if arr_to_save_npy is None:
                arr_to_save_npy = arr_to_save[idx_txt]
                idx_txt +=1
            else:
                arr_to_save_npy = np.concatenate((arr_to_save_npy, arr_to_save[idx_txt]), axis=0)
                idx_txt +=1
                # print(f"True {idx_json}")
        else:
            if arr_to_save_npy is None:
                arr_to_save_npy = arr_for_empty
            else:
                arr_to_save_npy = np.concatenate((arr_to_save_npy, arr_to_save[idx_txt]), axis=0)
                # print(f"Skip index {idx_json}")
        # print(arr_to_save_npy.shape)

        data_json_keyframes_period = data_json[idx_json].split("/")[2]
    arr_finnal_to_save_npy = np.concatenate((arr_finnal_to_save_npy, arr_to_save_npy), axis = 0)

    np.save(outfile, arr_finnal_to_save_npy)
except:
    np.save(outfile, arr_finnal_to_save_npy)
print(f"Saved {outfile}")
print("Len of arr", len(arr_finnal_to_save_npy))
f.close()
df_tx.close()