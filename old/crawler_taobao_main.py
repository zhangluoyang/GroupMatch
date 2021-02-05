"""
淘宝关键词爬虫
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 24, 2020
"""
from selenium import webdriver
from selenium.webdriver.common import keys
import time
import requests
import random
import os


# 创建浏览器
def functions(save_path: str, max_page_num: int, max_image_num: int):
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
            items = browser.find_elements_by_css_selector('.m-itemlist .items > div')
            for item in items:
                # 获取这张图片的下载地址
                img = item.find_element_by_css_selector(".pic-box .pic img").get_attribute("data-src")
                # 拼接完成的下载地址
                img_url = "http:" + img
                # print(img_url)
                # 通过requests下载这张图片
                sleep_time = random.randint(1, 2) + random.random()
                time.sleep(sleep_time)
                # 文件夹需要手动创建好
                file = open("{0}/{1}.jpg".format(save_path, image_index), "wb")
                file.write(requests.get(img_url).content)
                image_index += 1
                print("下载图片" + str(n))
                print("index:{0}".format(image_index))
                n += 1
                if image_index >= max_image_num:
                    break
            # 翻页操作
            browser.find_element_by_css_selector('.wraper:nth-last-child(1) .next > a').click()
            time.sleep(2)
            count += 1
            # 爬取 4 页内容
            if count == max_page_num:
                break
            if image_index >= max_image_num:
                break
    except Exception as e:
        print(e)


if __name__ == '__main__':
    key_words = ["学习桌"]
    style_words = ["欧式"]
    path = r"F:/taobao"
    # 每一种搜索的图片数目 2000
    max_image_num = 2000
    # 支持的最多翻页数目
    max_page_num = 100
    if not os.path.exists(path):
        os.mkdir(path)
    browser = webdriver.Chrome(r'../driver/chromedriver.exe')
    # 让浏览器打开淘宝
    browser.get("https://www.taobao.com/")
    # # 登录淘宝
    # # 切换成二维码登录
    # browser.find_element_by_xpath('//*[@id="login"]/div[1]/i').click()
    # # 判断当前页面是否为登录页面
    # while browser.current_url.startswith("https://login.taobao.com/"):
    #     print("等待用户输入")
    #     time.sleep(1)
    # print("登录成功!!! 并且休息10s钟")
    # time.sleep(1)
    # browser.get("https://www.taobao.com/")
    # for key_word in key_words:
    #     f_path = "{0}/{1}".format(path, key_word)
    #     if not os.path.exists(f_path):
    #         os.mkdir(f_path)
    #     for style_word in style_words:
    #         s_path = "{0}/{1}".format(f_path, style_word)
    #         if not os.path.exists(s_path):
    #             os.mkdir(s_path)
    #             # 清空搜索框
    #             browser.find_element_by_xpath('//*[@id="q"]').clear()
    #             # 搜索框输入关键词 并搜索内容`
    #             browser.find_element_by_xpath('//*[@id="q"]') \
    #                 .send_keys("{0} {1}".format(key_word, style_word), keys.Keys.ENTER)
    #             functions(s_path, max_page_num=max_page_num, max_image_num=max_image_num)
    #     time.sleep(random.random() * 10)
