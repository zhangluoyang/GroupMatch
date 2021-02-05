"""
一些必要的序列化操作
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 16, 2020
"""
import json
import cv2
import base64
import numpy as np


def image_to_base64(image_np: np.ndarray) -> str:
    """
    图片转换为 base64
    :param image_np:
    :return:
    """
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


def base64_to_image(base64_code: str) -> np.ndarray:
    """
    base64转换为图片 这个地方有些数据会报错。。
    :param base64_code:
    :return:
    """
    # https://blog.csdn.net/weixin_30549657/article/details/97705763 这里有分析原因
    # base64解码
    img = np.zeros(shape=(120, 120, 3), dtype=np.uint8)
    try:
        if len(base64_code) % 3 == 1:
            base64_code += "=="
        elif len(base64_code) % 3 == 2:
            base64_code += "="
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("error:{0}".format(e))
        print("base64 error:{0}".format(base64_code))
    finally:
        return img


def dumps_to_json_file(obj, file_path: str):
    json.dump(obj=obj, fp=open(file_path, "w", encoding="utf-8"))
