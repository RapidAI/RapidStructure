# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from rapid_layout import RapidLayout
from rapid_table import RapidTable


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    # 按照纵坐标（y）升序，横坐标（x）升序进行排序
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (
                abs(_boxes[j + 1][1] - _boxes[j][1]) < 10
                and _boxes[j + 1][0] < _boxes[j][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_boxes(layout_res: list):
    r_boxes = []
    # tmp_img = copy.deepcopy(img)
    for v in layout_res:
        bbox = np.round(v["bbox"]).astype(np.int32)
        label = v["label"]

        # start_point = (bbox[0], bbox[1])
        # end_point = (bbox[2], bbox[3])

        # cv2.rectangle(tmp_img, start_point, end_point, (0, 255, 0), 2)
        # box = [bbox[0], bbox[1], bbox[2], bbox[3]]
        if label == "table":
            r_boxes.append(bbox)

    # r_boxes = sorted_boxes2(r_boxes)
    dt_boxes = np.array(r_boxes)
    dt_boxes = np.array(sorted_boxes(dt_boxes))
    print(dt_boxes)
    return dt_boxes
    # print(r_boxes)
    # return r_boxes


def get_crop_img_list(img, dt_boxes):
    # 遍历dt_boxes列表
    crop_imgs = []
    for box in dt_boxes:
        x0, y0, x1, y1 = box
        # 从原始图像中截取指定位置的图像
        cropped_img = img[y0:y1, x0:x1]
        crop_imgs.append(cropped_img)

    return crop_imgs


def test_input():
    layout_engine = RapidLayout()

    cur_dir = Path(__file__).resolve().parent
    test_file_dir = cur_dir / "tests" / "test_files"
    img_path = test_file_dir / "layout.png"

    img = cv2.imread(str(img_path))
    # layout_res, elapse = layout_engine(img)
    layout_res, elapse = layout_engine(img)
    print(layout_res)
    dt_boxes = get_boxes(layout_res)
    img_crop_list = get_crop_img_list(img, dt_boxes)
    # 打印截取的图像列表
    # for i, cropped_img in enumerate(img_crop_list):
    #     cv2.imshow(f"Cropped Image {i + 1}", cropped_img)

    table_engine = RapidTable()
    ocr_engine = RapidOCR()

    # img_path = "tests/test_files/table.jpg"
    table_html = []
    for i, cropped_img in enumerate(img_crop_list):
        ocr_result, _ = ocr_engine(cropped_img)
        table_html_str, _ = table_engine(cropped_img, ocr_result)
        table_html.append(table_html_str)  # i,

    print(table_html)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # assert len(layout_res) == 13


if __name__ == "__main__":
    test_input()
