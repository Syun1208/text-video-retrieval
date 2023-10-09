import json 
import os
import tqdm

if __name__ == "__main__":
    dict_clip = '/home/hoangtv/Desktop/Long/text-video-retrieval/data/dicts/keyframes_id.json'
    save_plip = open('/home/hoangtv/Desktop/Long/text-video-retrieval/data/dicts/keyframes_id_plip.json', 'w')
    dict_plip= {}
    list_plip = []
    with open(dict_clip, 'r') as f:
        dict_id2cimg_clip = json.loads(f.read())
    print(dict_id2cimg_clip['0'])
    print(dict_id2cimg_clip['1'])
    # for value in tqdm.tqdm(list(dict_id2cimg_clip.values())[:10], desc='Converting'):
    #     list_shot_path = value['list_shot_path']
    #     for shot in list_shot_path:
    #         shot_id = shot['shot_id']
    #         shot_path = os.path.join('/home/hoangtv/Desktop/Long/text-video-retrieval/data/news', '/'.join(shot['shot_path'].split('/')[1:]))
    #         dict_plip = {"id": shot_id, "file_path": shot_path}
    #         list_plip.append(dict_plip)
    # save_plip.write(json.dumps(list_plip))
