## Rapid Structure
<p align="left">
    <a href="https://swhl-rapidstructuredemo.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Online Demo-blue"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<=3.11-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pepy.tech/project/rapid-layout"><img src="https://static.pepy.tech/personalized-badge/rapid-layout?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-layout"></a>
    <a href="https://pepy.tech/project/rapid-orientation"><img src="https://static.pepy.tech/personalized-badge/rapid-orientation?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-orientation"></a>
    <a href="https://pepy.tech/project/rapid-table"><img src="https://static.pepy.tech/personalized-badge/rapid-table?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=rapid-table"></a>
</p>

- 🎉🎉🎉 推出知识星球[RapidAI私享群](https://t.zsxq.com/0duLBZczw)，这里的提问会优先得到回答和支持，也会享受到RapidAI组织后续持续优质的服务，欢迎大家的加入。
- 该部分的功能主要针对文档类图像，包括文档图像分类、版面分析和表格识别。
- 可配套使用项目：[RapidOCR](https://github.com/RapidAI/RapidOCR)

### [文档方向分类](./docs/README_Orientation.md)
### [版面分析](./docs/README_Layout.md)
### [表格识别](./docs/README_Table.md)

### 整体流程
```mermaid
flowchart LR
    A[/文档图像/] --> B(文档方向分类 rapid_orientation) --> C(版面分析 rapid_layout) & D(表格识别 rapid_table) --> E(OCR识别 rapidocr_onnxruntime)
    E --> F(结构化输出)
```
