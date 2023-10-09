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

team_3_dict = os.path.join(ROOT, 'keyframes_id_all.json')
with open(team_3_dict, 'r') as file_team_3:
    data_team_3 = json.loads(file_team_3.read())
team_1_dict = os.path.join(ROOT, 'keyframes_id_search.json')
with open(team_1_dict, 'r') as file_team_1:
    data_team_1 = json.loads(file_team_1.read())
def main():
    
    
    length_folder = len(list(set(map(lambda x: '/'.join(x.split('/')[:2]), data_team_1))))

    for i in range(length_folder):
        item = str(i+1)
        if len(str(i+1)) == 1:
            item = '0' + str(i+1)
        lenght_team_3 = list(filter(lambda x: "Data/Keyframes_L" + item + "/" in x, list(data_team_3.values())))
        list_LO = list(map(lambda x: '/'.join(x.split('/')[:3]), list(lenght_team_3)))
        list_LO = list(set(list_LO))
        print('Duplicate folder Keyframes_L{} team 3: {}'.format(item, len(lenght_team_3) - len(list(set(lenght_team_3)))))
        lenght_team_1 = list(filter(lambda x: "data/Keyframes_L"+ item +"/" in x, list(data_team_1)))
        print('Duplicate folder Keyframes_L{} team 1: {}'.format(item, len(lenght_team_1) - len(list(set(lenght_team_1)))))
        print('Length folder Keyframes_L{} team 3: {} '.format(item, len(list(lenght_team_3))))
        print('Length folder Keyframes_L{} team 1: {} '.format(item, len(list(lenght_team_1))))
        print('Correlation among 2 teams: ', len(list(lenght_team_1)) == len(list(lenght_team_3)))
        print("=====================================")
if __name__ == '__main__':
    main()