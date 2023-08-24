# Quá trình xử lý OCR

## Đầu vào `thư mục chứa các ảnh` và đầu ra là `txt file`
- Cấu trúc txt file (Example)

```
L01_V001,000182, htv hd 06:30:36 giây
L01_V001,000251, hmx hd 06:30:38 yến phan ny như giây
L01_V001,000321, hiva hd 06:30:41 giây
L01_V001,000377, 79 h hd 06:30:43 ống wurld đông laius day ...
...
```

### Cú pháp chạy
```
cd CRAFT-pytorch
pip install -r requirements.txt
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```

### Ví dụ
```
python test.py --trained_model=craft_mlt_25k.pth --test_folder=/home/toonies/AI-Challenge/AI_Challenge/data/Keyframes_L01-001/Keyframes_L01/L01_V001
```


## Convert from `*.txt` $\rightarrow$ `*.npy` $\rightarrow$ `*.bin`
