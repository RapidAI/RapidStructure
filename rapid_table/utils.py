# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

InputType = Union[str, np.ndarray, bytes, Path]


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 4:
            return self.cvt_four_to_three(img)

        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → RGB"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


def vis_table(table_res: str, save_path: str) -> None:
    style_res = """<style>td {border-left: 1px solid;border-bottom:1px solid;}
                   table, th {border-top:1px solid;font-size: 10px;
                   border-collapse: collapse;border-right: 1px solid;}
                </style>"""
    prefix_table, suffix_table = table_res.split("<body>")
    new_table_res = f"{prefix_table}{style_res}<body>{suffix_table}"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(new_table_res)
    print(f"The infer result has saved in {save_path}")
