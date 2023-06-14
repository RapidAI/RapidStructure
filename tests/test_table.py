# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import pytest

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from rapid_table import RapidTable

rapid_table = RapidTable()

test_file_dir = cur_dir / 'test_files'
img_path = str(test_file_dir / 'table.jpg')
img = cv2.imread(img_path)


@pytest.mark.parametrize(
    'img_content',
    [
        img_path,
        str(img_path),
        open(img_path, 'rb').read(),
        img
    ]
)
def test_multi_input(img_content):
    table_html_str, elapse = rapid_table(img_content)
    assert table_html_str.count('<tr>') == 16
