# !apt-get install ffmpeg
# !pip install ffmpeg-python pillow
# !git clone https://github.com/soCzech/TransNetV2.git
# %cd TransNetV2
# !git lfs fetch https://github.com/soCzech/TransNetV2.git
# !git lfs checkout



import os
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
import cv2
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.abspath(ROOT))
sys.path.append(os.path.join(ROOT, "data_processing/TransNetV2"))

WORK_DIR = os.path.dirname(ROOT)

video_paths = sorted(glob.glob(os.path.join(WORK_DIR,"data/videos/Keyframes_L28/*.mp4"))) # Edit
des_path = f"{WORK_DIR}/data/Keyframes_L28/" # Edit
class TransNetV2:

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")

        self._input_size = (27, 48, 3)
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                          f"Re-download them manually and retry. For more info, see: "
                          f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 8 #25
            no_padded_frames_end = 8 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                                all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                                      "individual frames from video file. Install `ffmpeg` command line tool and then "
                                      "install python wrapper by `pip install ffmpeg-python`.")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))
        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        from PIL import Image, ImageDraw

        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25

        # pad frames so that length of the video is divisible by width
        # pad frames also by len(predictions) pixels in width in order to show predictions
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(np.split(
            np.concatenate(np.split(img, height), axis=2)[0], width
        ), axis=2)[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        # iterate over all frames
        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

            # we can visualize multiple predictions per single frame
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255

                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img
    
model = TransNetV2(os.path.join(ROOT, "data_processing/TransNetV2/inference/transnetv2-weights"))

for video_path in video_paths[21:]:
    folder_name = video_path.split('/')[-1].replace( '.mp4','')

    # test_folder = int(folder_name.replace('L05_V',''))
    # print(test_folder)
    # if test_folder <= 244 or test_folder==253 or test_folder==271:
    #     print(f"Skip {folder_name}")
    #     continue

    folder_path = des_path + f'{folder_name}'
    os.mkdir(folder_path)

    #transnetV2
    video_frames, single_frame_predictions, all_frame_predictions = \
      model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)
    with open(f"{folder_path}.txt", 'w') as f:
        for sc in scenes:
            f.write(str(sc) + '\n')

    cam = cv2.VideoCapture(video_path)
    currentframe = 0
    index = 0

    while True:
        ret,frame = cam.read()
        if ret:
            currentframe += 1
            # for sc in scenes:
            if (index>len(scenes)-1):
              break
            idx_first = int(scenes[index][0])
            idx_end = int(scenes[index][1])
            idx_025 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/4)
            idx_05 = int(scenes[index][0] + (scenes[index][1]-scenes[index][0])/2)
            idx_075 = int(scenes[index][0] + 3*(scenes[index][1]-scenes[index][0])/4)

            #### First ####
            if currentframe - 1 == idx_first:
                filename_first = "{}/{:0>6d}.jpg".format(folder_path, idx_first)
                # video_save = cv2.resize(video[idx_first], (1280,720))
                cv2.imwrite(filename_first, frame)

            # #### End ####
            if currentframe - 1 == idx_end:
                filename_end = "{}/{:0>6d}.jpg".format(folder_path, idx_end)
                # video_save = cv2.resize(video[idx_end], (1280,720))
                cv2.imwrite(filename_end, frame)
                index += 1

            #### 025 ####
            if currentframe - 1 == idx_025:
                filename_025 = "{}/{:0>6d}.jpg".format(folder_path, idx_025)
                # video_save = cv2.resize(video[idx_025], (1280,720))
                cv2.imwrite(filename_025, frame)

            # #### 05 ####
            if currentframe - 1 == idx_05:
                filename_05 = "{}/{:0>6d}.jpg".format(folder_path, idx_05)
                # video_save = cv2.resize(video[idx_05], (1280,720))
                cv2.imwrite(filename_05, frame)

            # #### 075 ####
            if currentframe - 1 == idx_075:
                filename_075 = "{}/{:0>6d}.jpg".format(folder_path, idx_075)
                # video_save = cv2.resize(video[idx_075], (1280,720))
                cv2.imwrite(filename_075, frame)

        else:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(len(scenes))
    print("---------------------------------------")



