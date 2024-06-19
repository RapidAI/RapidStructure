# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import cv2

from rapid_layout import RapidLayout, VisLayout
from rapid_orientation import RapidOrientation
from rapid_table import RapidTable, VisTable


def vis_table(table_res):
    style_res = """<style>td {border-left: 1px solid;border-bottom:1px solid;}
                   table, th {border-top:1px solid;font-size: 10px;
                   border-collapse: collapse;border-right: 1px solid;}
                </style>"""
    prefix_table, suffix_table = table_res.split("<body>")
    new_table_res = f"{prefix_table}{style_res}<body>{suffix_table}"

    draw_img_save = Path("./inference_results/")
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True, exist_ok=True)

    html_path = str(draw_img_save / "table_result.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_table_res)
    print(f"The infer result has saved in {html_path}")


def demo_layout():
    layout_engine = RapidLayout(box_threshold=0.5, model_type="pp_layout_cdla")

    img_path = "tests/test_files/layout.png"
    img = cv2.imread(img_path)

    boxes, scores, class_names, *elapse = layout_engine(img)
    ploted_img = VisLayout.draw_detections(img, boxes, scores, class_names)
    if ploted_img is not None:
        cv2.imwrite("layout_res.png", ploted_img)


def demo_table():
    from rapidocr_onnxruntime import RapidOCR, VisRes

    ocr_engine = RapidOCR()
    vis_ocr = VisRes()
    table_engine = RapidTable()
    viser = VisTable()

    img_path = "1.png"
    ocr_result, _ = ocr_engine(img_path)

    table_html_str, table_cell_bboxes, _ = table_engine(img_path, ocr_result)

    boxes, txts, scores = list(zip(*ocr_result))

    save_dir = Path("./inference_results/")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    save_html_path = save_dir / f"{Path(img_path).stem}.html"
    save_drawed_path = save_dir / f"vis_{Path(img_path).name}"
    vis_imged = viser(
        img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path
    )
    res = vis_ocr(vis_imged, boxes)
    cv2.imwrite("only_vis_det.png", res)
    print(table_html_str)


def demo_orientation():
    orientation_engine = RapidOrientation()
    img = cv2.imread("tests/test_files/img_rot180_demo.jpg")
    cls_result, _ = orientation_engine(img)
    print(cls_result)


if __name__ == "__main__":
    demo_layout()
    # demo_table()
    # demo_orientation()
