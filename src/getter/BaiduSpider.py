# #
import os
import re
import requests
import random
from urllib.parse import urlencode
from src.Logger import GeneralLogger
from src.getter.Spider import Spider
from requests.adapters import HTTPAdapter
from threading import Thread
from multiprocessing import Process, Value
from multiprocessing import Queue as Q
from queue import Queue
from src.Variables import *

# # Hyper Parameters
THREAD_NUM = 10


# # Baidu Image Spider

class BaiduSpider(Spider):
    def __init__(self, first_word="", big_label=''):
        super(BaiduSpider, self).__init__(first_word)
        self.big_label = big_label
        self.list_num = 20
        self.query_params = {
            "tn": "baiduimage",
            "ie": "utf-8",
            "word": self.last_input,
            "ct": "201326592",
            "v": "flip",
            "pn": 0
        }
        self.count = 0

    def get_list_num(self, retry=3):
        global BaiduSpiderLogger
        url = "http://image.baidu.com/search/flip?" + urlencode(self.query_params)
        try:
            self.response = requests.get(url)
            if self.response.status_code != 200:
                return False
            pattern = re.compile("""flip.setData\('imgData',.*?"listNum":(.*?),.*?"data":\[.*?\].*?""", re.S)
            res = pattern.findall(self.response.text)
            if len(res) > 0:
                self.list_num = int(res[0].strip())
                print(self.list_num)
                return True
            else:
                back_file_name = LOGS_ROOT + "/response_bak_" + str(random.randint(100000, 999999)) + ".html"
                with open(back_file_name, "w", encoding="utf-8") as f:
                    f.write(self.response.text)
                    if 'BaiduSpiderLogger' in globals():
                        BaiduSpiderLogger.warning(
                            "Parse Response Failed, No Urls Found, Additional:" + back_file_name)
                    else:
                        print("Warning: Parse Response Failed, No Urls Found, Additional:" + back_file_name)
                return False
        except Exception as e:
            if 'BaiduSpiderLogger' in globals():
                BaiduSpiderLogger.error(
                    "Get List Num Error Because of " + str(e.__class__) + " " + str(e))
            else:
                print("ERROR: Get List Num Error Because of " + str(e.__class__) + " " + str(e))
            print("retrying, last chance:", retry)
            if retry <= 0:
                return False
            else:
                self.get_list_num(retry - 1)

    def get_response(self, page, queue):
        # # 获取响应
        if page > self.list_num:
            return
        else:
            self.query_params['pn'] = page
        url = "http://image.baidu.com/search/flip?" + urlencode(self.query_params)
        print("From", url)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # with open("../../tmp/temp.html", "w", encoding="utf-8") as f:
                #     f.write(response.text)
                self.parse_response(queue, response.text)

            else:
                if 'BaiduSpiderLogger' in globals():
                    BaiduSpiderLogger.warning(
                        "Get Response Failed From " + url + ",response status code:" + str(self.response.status_code))
                else:
                    print("Warning: Get Response Failed From " + url + ",response status code:" + str(
                        self.response.status_code))
        except Exception as e:
            if 'BaiduSpiderLogger' in globals():
                BaiduSpiderLogger.error(
                    "Get Response Failed From " + url + "Because of" + str(e.__class__) + " " + str(e))
            else:
                print("ERROR: Get Response Failed From " + url + "Because of" + str(e.__class__) + " " + str(e))

    def parse_response(self, queue, text):
        pattern = re.compile("""flip.setData\('imgData',.*?"listNum":.*?,.*?"data":\[(.*?)\].*?""", re.S)
        res = pattern.findall(text)
        if res is not None:
            pattern_1 = re.compile(""""objURL":"(.*?)",""", re.S)
            res_1 = set(pattern_1.findall(res[0]))
            for ele in res_1:
                queue.put(ele)
        else:
            back_file_name = LOGS_ROOT + "/response_bak_" + str(random.randint(100000, 999999)) + ".html"
            with open(back_file_name, "w", encoding="utf-8") as f:
                f.write(text)
                if 'BaiduSpiderLogger' in globals():
                    BaiduSpiderLogger.warning(
                        "Parse Response Failed, No Urls Found, Additional:" + back_file_name)
                else:
                    print("Warning: Parse Response Failed, No Urls Found, Additional:" + back_file_name)

    def write_to_local(self, num, queue, _base):
        s = requests.Session()
        s.mount('http://', HTTPAdapter(max_retries=2))
        s.mount('https://', HTTPAdapter(max_retries=2))
        while queue.qsize() > 0:
            ele = queue.get()
            tmp_pos = ele.find('@')
            if tmp_pos != -1:
                ele = ele[: tmp_pos]
            try:
                response = s.get(ele, timeout=3)
                if response.status_code == 200:
                    # ####################################################3
                    tmp_pos = ele.find('?')
                    if tmp_pos != -1:
                        ele = ele[tmp_pos:]
                    tmp_pos = ele.rfind('.')
                    ex_name = '.jpeg'
                    if tmp_pos != -1:
                        ex_name = ele[tmp_pos:]
                    if '/' in ex_name:
                        ex_name = '.jpeg'
                    # ######################################################3
                    if hasattr(num, 'value'):
                        num = num.value
                    if hasattr(_base, 'value'):
                        _base = _base.value
                    with open(
                            IMAGES_ROOT + "/" + self.big_label + "/" + self.last_input + "/" + self.last_input + "_" +
                            str(num) + "_" + str(self.count) + "_" + str(_base) + ex_name, "wb") as f:
                        f.write(response.content)
                        print("DownLoad From: ", ele, " 😀 ")
                else:
                    print("DownLoad From:  " + ele, " 😡 Failed")
                self.count += 1
            except Exception as e:
                if 'BaiduSpiderLogger' in globals():
                    BaiduSpiderLogger.error(
                        "Get Response Failed From " + ele + " Because of " + str(e.__class__) + " " + str(e))
                else:
                    pass
                    # print()
                    # print("ERROR: Get Response Failed From " + ele + " Because of " + str(e.__class__) + " " + str(e))
                self.write_to_local(num, queue, _base)

    def run_spider(self, _base=0, is_thread=True):
        new_dir = IMAGES_ROOT + "/" + self.big_label + "/" + self.last_input
        # new_dir = os.path.join(IMAGES_ROOT, self.last_input)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        if is_thread:
            shared_queue = Queue()
        else:
            shared_queue = Q()
        global THREAD_NUM
        for i in range(THREAD_NUM):
            self.get_response(_base + i * 60, shared_queue)
        print("本次找到%d张图片:" % shared_queue.qsize())
        if is_thread:
            threads = []
            for i in range(THREAD_NUM):
                threads.append(Thread(target=self.write_to_local, args=(i, shared_queue, _base)))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            processes = []
            for i in range(THREAD_NUM):
                processes.append(
                    Process(target=self.write_to_local, args=(Value('i', i), shared_queue, Value('i', _base))))
            for process in processes:
                process.start()
            for process in processes:
                process.join()


def baidu_spider(word, big_label):
    BaiduSpiderLogger = GeneralLogger(path=LOGS_ROOT + '/baidu_spider.log')
    print("BeGin..., word: " + word)
    bs = BaiduSpider(word, big_label)
    if bs.get_list_num():
        bs.run_spider(0, is_thread=False)
    else:
        print("获取列表失败")
