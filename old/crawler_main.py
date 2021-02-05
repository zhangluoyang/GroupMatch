import re
import os
import cv2
import urllib
import numpy as np
import urllib.request
from urllib.parse import quote

"""
简单的爬虫程序
@author zhangluoyang
"""


class BaiduImage(object):
    """
     百度网络爬虫
    """

    def __init__(self, keyword, count=512, save_path="img", rn=60):
        self.keyword = keyword
        self.count = count
        self.save_path = save_path
        self.rn = rn

        self.index = 0
        self.__imageList = []
        self.__totleCount = 0
        self.__encodeKeyword = quote(self.keyword)
        self.__acJsonCount = self.__get_ac_json_count()

        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36"

        self.headers = {'User-Agent': self.user_agent, "Upgrade-Insecure-Requests": 1,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Encoding": "gzip, deflate, sdch",
                        "Accept-Language": "zh-CN,zh;q=0.8,en;q=0.6",
                        "Cache-Control": "no-cache"}

    def search(self):
        for i in range(0, self.__acJsonCount):
            # 获得url数据
            url = self.__get_search_url(i * self.rn)
            # 根据url获得响应数据
            res = str(self.__get_response(url), encoding="utf-8")
            response = res.replace("\\", "")
            # 获得所有的图片列表数据
            image_url_list = self.__pick_image_urls(response)
            # 存储图片数据到本地
            self.__save(image_url_list)

    def __save(self, image_url_list):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for i in range(len(image_url_list)):
            imageUrl = image_url_list[i]
            host = self.get_url_host(imageUrl)
            self.headers["Host"] = host
            try:
                resp = urllib.request.urlopen(imageUrl)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                cv2.imwrite("{0}/{1}.jpg".format(self.save_path, self.index), image)
                self.index = self.index + 1
                self.__totleCount += 1
            except Exception as e:
                print("Exception" + str(e))
        print("已存储 " + str(self.__totleCount) + " 张图片")

    def __pick_image_urls(self, response):
        reg = r'"thumbURL":"(https://.*?.jpg)"'
        imgre = re.compile(reg)
        imglist = re.findall(imgre, response)
        return imglist

    def __get_response(self, url):
        page = urllib.request.urlopen(url)
        return page.read()

    def __get_search_url(self, pn):
        return "http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=" + self.__encodeKeyword + "&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word=" + self.__encodeKeyword + "&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&pn=" + str(
            pn) + "&rn=" + str(self.rn) + "&gsm=1000000001e&1486375820481="

    def get_url_host(self, url):
        reg = r'http://(.*?)/'
        hostre = re.compile(reg)
        host = re.findall(hostre, url)
        if len(host) > 0:
            return host[0]
        return ""

    def __get_ac_json_count(self):
        a = self.count % self.rn
        c = self.count / self.rn
        if a:
            c += 1
        return int(c)


if __name__ == "__main__":
    key_words = ["bed", "sofa", "tv_cabinet", "dining_table", "study_table", "book_cabinet", "entrance_cabinet",
                 "bath_room_cabinet", "toilte", "shower", "cloth_cabinet", "kitchen_cabinet"]
    style_words = ["china", "american", "european", "modern"]

    china_key_words = ["床", "沙发", "电视柜", "餐桌", "学习桌", "书柜", "玄关柜",
                       "卫浴柜", "马桶", "花洒", "衣柜", "橱柜"]
    china_style_words = ["中式", "美式", "欧式", "现代"]

    path = r"D:/workspace/images"
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(key_words)):
        key_word = key_words[i]
        f_path = "{0}/{1}".format(path, key_word)
        if not os.path.exists(f_path):
            os.mkdir(f_path)
        for j in range(len(style_words)):
            style_word = style_words[j]
            s_path = "{0}/{1}".format(f_path, style_word)
            if not os.path.exists(s_path):
                os.mkdir(s_path)
                search = BaiduImage("{0}+{1}".format(china_key_words[i], china_style_words[j]), save_path=s_path, rn=60)
                search.search()
