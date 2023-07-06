## rapid-orientation
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapid-orientation/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-orientation"></a>
    <a href="https://pepy.tech/project/rapid-orientation"><img src="https://static.pepy.tech/personalized-badge/rapid-orientation?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>


### 1. Install package by pypi.
```bash
$ pip install rapid-orientation
```

### 2. Run by script.
- RapidOrientation has the default `model_path` value, you can set the different value of `model_path` to use different models, e.g. `orientation_engine = RapidOrientation(model_path='rapid_orientation.onnx')`
- See details, for [README_Layout](https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Orientation.md) .
- 📌 `layout.png` source: [link](https://github.com/RapidAI/RapidStructure/blob/main/test_images/layout.png)

```python
from rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation()

with open('test_images/layout.png', 'rb') as f:
    img = f.read()
orientation_res, elapse = orientation_engine(img)
print(orientation_res)
```

### 3. Run by command line.
- Usage:
    ```bash
    $ rapid_orientation -h
    usage: rapid_orientation [-h] -img IMG_PATH [-m MODEL_PATH]

    optional arguments:
    -h, --help            show this help message and exit
    -img IMG_PATH, --img_path IMG_PATH
                        Path to image for layout.
    -m MODEL_PATH, --model_path MODEL_PATH
                        The model path used for inference
    ```
- Example:
    ```bash
    $ rapid_orientation -img layout.png
    ```

### 4. Result.
```python
# Return str, four types:：0 | 90 | 180 | 270
```