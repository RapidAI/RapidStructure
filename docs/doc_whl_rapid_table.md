## rapid-table
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapid-table/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-table"></a>
    <a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>


### 1. Install package by pypi.
⚠️Attention: After `rapid_table>=v0.1.0`, you need to install `rapidocr_onnxruntime` package firstly.
```bash
pip install rapidocr_onnxruntime
pip install rapid-table
```

### 2. Run by script.
- RapidTable has the default `model_path` value, you can set the different value of `model_path` to use different models, e.g. `table_engine = RapidTable(model_path='ch_ppstructure_mobile_v2_SLANet.onnx')`
- See details, for [README_Table](https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Table.md) .
- 📌 `table.jpg` source: [link](https://github.com/RapidAI/RapidStructure/blob/main/test_images/table.jpg)

    ````python
    from rapid_table import RapidTable
    from rapidocr_onnxruntime import RapidOCR

    table_engine = RapidTable()
    ocr_engine = RapidOCR()

    img_path = 'test_images/table.jpg'

    ocr_result, _ = ocr_engine(img_path)
    table_html_str, _ = table_engine(img_path, ocr_result)

    print(table_html_str)
    ````

### 3. Run by command line.
- Usage:
    ```bash
    $ rapid_table -h
    usage: rapid_table [-h] [-v] -img IMG_PATH [-m MODEL_PATH]

    optional arguments:
    -h, --help            show this help message and exit
    -v, --vis             Wheter to visualize the layout results.
    -img IMG_PATH, --img_path IMG_PATH
                        Path to image for layout.
    -m MODEL_PATH, --model_path MODEL_PATH
                        The model path used for inference.
    ```

- Example:
    ```bash
    $ rapid_table -v -img test_images/table.jpg
    ```

### 4. Result.
- Return value.
    ```html
    <html><body><table><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></table></body></html>
    ```
- Visualize result.
    <div align="center">
        <table><tr><td>Methods</td><td></td><td></td><td></td><td>FPS</td></tr><tr><td>SegLink [26]</td><td>70.0</td><td>86d><td.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td></td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2</td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.87.4</td><td>81.7</td><td></td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td></td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><><ttd></td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN[16]</td><td>79</td><td>88</td><td>83</td><td></td></tr><tr><td>ATRR</>[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td></td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</t91/d><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG[41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td></td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85<t..40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></table>
    </div>

### For details, see [Rapid Table](https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Table.md)