import json

path_txt = "/mnt/HDD6/tuong/Nhan_CDT/ASR/asr.txt"
path_json = "/mnt/HDD6/tuong/Nhan_CDT/ASR/keyframes.json"
path_txt_out = "/mnt/HDD6/tuong/Nhan_CDT/ASR/no_hope_asr.txt"


def str_to_int(frames: str):
    return int(frames.split("/")[-1].replace(".jpg",""))


with open(path_txt, "r", encoding='utf-8') as txt_file:
    lines = txt_file.readlines()

data_js = open(path_json)
info_js = json.load(data_js)


idx_js = 0
with open(path_txt_out, "w") as write_txt:
    for line in lines:
        
        parts = line.strip().split(',')

        p_start = parts[0] # L01_V001
        content = "".join(parts[3:])
        from_frames = int(parts[1])
        to_frames = int(parts[2])

        break_while = 1
        
        while break_while:
            num_frames = str_to_int(info_js[idx_js])
            if  num_frames <= to_frames and info_js[idx_js].split("/")[-2] == p_start:
                print("True")
                write_txt.write(f"{p_start},{num_frames},{content}" + "\n")
                idx_js +=1
            
            elif (to_frames < num_frames) or (info_js[idx_js].split("/")[-2] != p_start) :
                break_while = 0

data_js.close()