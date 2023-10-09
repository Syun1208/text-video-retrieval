import pandas as pd
import os

submission = './submit_back'
submission_main = './submission_back'


os.makedirs(submission_main, exist_ok=True)

for file in sorted(os.listdir(submission)):
    list_change = []
    data = pd.read_csv(os.path.join(submission, file))

    first_row = data.iloc[0]
    shuffled_df = data.iloc[1:].sample(frac=1).reset_index(drop=True).astype(str)

    # Kết hợp hàng đầu tiên và DataFrame đã xáo đổi
    result_df = pd.concat([first_row.to_frame().T, shuffled_df], ignore_index=True)
    for value in result_df.iloc[:, 1].values.tolist():
        if len(str(value)) == 4:
            value = "00" + str(value)
        else:
            value = "0" + str(value)
        list_change.append(value)
    result_df.iloc[:, 1] = list_change

    result_df.to_csv(os.path.join(submission_main, file), index=False)

