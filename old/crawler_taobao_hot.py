"""
淘宝热卖网爬虫
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 29, 2020
"""
from selenium import webdriver
from selenium.webdriver.common import keys
import time
import requests
import numpy as np
import cv2
import random
import os
import urllib


# 创建浏览器
def functions(save_path: str, max_page_num: int, max_image_num: int, key_word: str, style_word: str):
    """
    :param save_path: 存储路径
    :param max_page_num: 最大翻页数目
    :param max_image_num: 最大图片数目
    :return:
    """
    n = 1
    count = 1
    image_index = 0
    try:
        while True:
            elements = browser.find_element_by_id('J_u_root') \
                .find_element_by_class_name("pc-search-items-list") \
                .find_elements_by_xpath("li")
            for item in elements:
                print(item.text)
                print(key_word in item.text and style_word in item.text)
                if key_word in item.text and style_word in item.text:
                    # 获取这张图片的下载地址
                    time.sleep(random.randint(1, 2) + random.random())
                    img_url = item.find_element_by_class_name("pc-items-item-a").find_element_by_css_selector(
                        "img").get_attribute("src")
                    if img_url is None:
                        print("img_url 找不到 过滤词条数据")
                        item.find_element_by_class_name("pc-items-item-a")
                        img_url = item.find_element_by_class_name("pc-items-item-a").find_element_by_css_selector(
                            "img").get_attribute("src")
                    if img_url is None:
                        continue
                    # 文件夹需要手动创建好
                    img_url = img_url.replace("https", "http")
                    print(img_url)
                    file = open("{0}/{1}.jpg".format(save_path, image_index), "wb")
                    file.write(requests.get(img_url).content)
                    image_index += 1
                    print("下载图片" + str(n))
                    print("index:{0}".format(image_index))
                    n += 1
                    if image_index >= max_image_num:
                        break
            # 翻页操作
            browser.find_element_by_id('J_pc-search-page-nav').click()
            time.sleep(1)
            count += 1
            if count == max_page_num:
                break
            if image_index >= max_image_num:
                break
    except Exception as e:
        print(e)


if __name__ == "__main__":
    key_words = ["书桌"]
    style_words = ["欧式"]
    path = r"F:/taobao"
    # 每一种搜索的图片数目 2000
    max_image_num = 2000
    # 支持的最多翻页数目
    max_page_num = 100
    if not os.path.exists(path):
        os.mkdir(path)
    browser = webdriver.Chrome(r'../driver/chromedriver.exe')
    browser.get("https://uland.taobao.com/sem/tbsearch")
    for key_word in key_words:
        f_path = "{0}/{1}".format(path, key_word)
        if not os.path.exists(f_path):
            os.mkdir(f_path)
        for style_word in style_words:
            s_path = "{0}/{1}".format(f_path, style_word)
            if not os.path.exists(s_path):
                os.mkdir(s_path)
            browser.find_element_by_id("J_search_key").clear()
            time.sleep(random.random() * 2)
            browser.find_element_by_id("J_search_key").send_keys(
                "{0}".format("{0} {1}".format(key_word, style_word)), keys.Keys.ENTER)
            functions(s_path, max_page_num=max_page_num,
                      max_image_num=max_image_num,
                      key_word=key_word,
                      style_word=style_word)
        time.sleep(random.random() * 1)
