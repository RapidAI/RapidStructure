<div align="center">
  <div align="center">
    <h1><b>📃 Rapid Structure</b></h1>
  </div>

<a href="https://swhl-rapidstructuredemo.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Online Demo-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://pepy.tech/project/rapid-layout"><img src="https://static.pepy.tech/personalized-badge/rapid-layout?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-layout"></a>
<a href="https://pepy.tech/project/rapid-orientation"><img src="https://static.pepy.tech/personalized-badge/rapid-orientation?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-orientation"></a>
<a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-table"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

### 简介
该部分的功能主要针对文档类图像，包括文档图像分类、版面分析和表格识别。

可配套使用项目：[RapidOCR](https://github.com/RapidAI/RapidOCR)

相关模型托管：[Hugging Face Models](https://huggingface.co/SWHL/RapidStructure)

### [文档方向分类](./docs/README_Orientation.md)
### [版面分析](https://github.com/RapidAI/RapidLayout)
### [表格识别](./docs/README_Table.md)
更多表格识别：[TableStructureRec](https://github.com/RapidAI/TableStructureRec)

### 🔥🔥[版面还原](https://github.com/RapidAI/RapidLayoutRecover)

### 整体流程
```mermaid
flowchart TD
    A[/文档图像/] --> B([文档方向分类 rapid_orientation]) --> C([版面分析 rapid_layout])
    C --> D([表格识别 rapid_table]) & E([公式识别 rapid_latex_ocr]) & F([文字识别 rapidocr_onnxruntime]) --> G([版面还原 rapid_layout_recover])
    G --> H[/结构化输出/]
```
