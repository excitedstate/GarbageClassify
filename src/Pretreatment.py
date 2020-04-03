# # Imports
import os
from random import shuffle
import shutil
import imghdr
import numpy as np
from cv2 import cv2
from skimage.metrics import structural_similarity
from src.getter.BaiduSpider import baidu_spider
from src.Variables import *


def cv_show(img, win_name="test", t=0):
    print("showing image(%dx%d) in %s" % (img.shape[0], img.shape[1], win_name))
    cv2.imshow(win_name, img)
    cv2.waitKey(t)
    cv2.destroyAllWindows()
    print("Over!")


class PreTreatment:
    def __init__(self, _folder_path, _class_name='test', need_check=True):
        self.folder_path = _folder_path
        self.class_name = _class_name
        if not self.folder_path.endswith("/"):
            self.folder_path += "/"
        if need_check:
            self._check()

    def _check(self):
        print(self.class_name, "自检开始:")
        count, not_ok = 1, 0
        file_list = os.listdir(self.folder_path)

        for file_name in file_list:
            full_name = os.path.join(self.folder_path, file_name)
            print('\r', count, " / ", len(file_list), end='')
            count += 1
            try:
                if cv2.imdecode(np.fromfile(full_name, dtype=np.uint8), -1) is None:
                    os.remove(full_name)
                    not_ok += 1
                    print(" 已删除: ", full_name)
                else:
                    temp = imghdr.what(full_name)
                    if temp is not None:
                        tmp_pos = file_name.rfind(".")
                        if tmp_pos != -1:
                            os.rename(full_name, os.path.join(self.folder_path, file_name[:tmp_pos + 1] + temp))
                        else:
                            os.remove(full_name)
            except Exception as e:
                os.remove(full_name)
                not_ok += 1
                print(" 已删除: ", full_name)
                print(e.__class__, str(e))
                continue
        print("\n删除了%d张无法打开的图片" % not_ok)
        print(self.class_name, "自检完成")

    def delete_doc_strict(self, threshold=0.5):
        print("开始去掉图片库%s中的文档或无背景图" % self.folder_path)
        count = 0
        for i, img_path in enumerate(os.listdir(self.folder_path)):
            try:
                img_path = self.folder_path + img_path
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                if img is not None:
                    he_ret = self.histogram_equalization(img)[1]
                    mean_ = np.mean([ele[2] for ele in he_ret])
                    if mean_ > threshold:
                        count += 1
                        os.remove(img_path)
                    print("\r共删除了%d张疑似文档的图片, 检查了%d张图片" % (count, i), end='')
            except Exception as e:
                # cv_show(img, win_name=img_path)
                # print(img_path)
                print(e.__class__, e.__str__())
                os.remove(img_path)  # # 读取不了的图片
        print()

    @staticmethod
    def histogram_equalization(img):
        channels = cv2.split(img)
        max_h = []
        channel_size = img.shape[0] * img.shape[1]
        for channel in channels:
            # # 可用 cv2.equalizeHist(chan) 函数代替
            # # 即 channel[:] = equalizeHist(channel)
            # # 计算直方图可以用cv2.calcHist() 函数代替 具体用法查询百度
            h_c = []
            for i in range(256):
                h_c.append(channel[channel == i].size)
            c_c = np.cumsum(h_c) / channel_size
            channel[:] = 255 * c_c[channel]
            max_h.append((np.argmax(h_c), np.max(h_c), np.max(h_c) / channel_size))
        return cv2.merge(channels), max_h

    @staticmethod
    def whiten(img: np.ndarray):
        # # 白化
        channels = cv2.split(img)
        new_channels = []
        for channel in channels:
            mean_ = np.mean(channel.ravel())
            std_ = np.std(channel.ravel())
            new_channels.append((channel - mean_) / std_)
        return cv2.merge(new_channels)

    @staticmethod
    def gray_world(img: np.ndarray):
        b, g, r = cv2.split(img)[:3]
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        k = (b_mean + g_mean + r_mean) / 3
        kb, kg, kr = k / b_mean, k / g_mean, k / r_mean
        return cv2.merge(
            (cv2.addWeighted(b, kb, 0, 0, 0), cv2.addWeighted(g, kg, 0, 0, 0), cv2.addWeighted(r, kr, 0, 0, 0)))

    def mirror(self, img, mode=0):
        ret = np.zeros(img.shape, dtype=np.uint8)
        if mode == 0:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    ret[i, img.shape[1] - j - 1] = img[i, j]
        elif mode == 1:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    ret[img.shape[0] - i - 1, j] = img[i, j]
        else:
            ret = self.mirror(self.mirror(img, 0), 1)
        return ret

    def spin(self, img, angle=1):
        if angle % 4 == 1:
            b, g, r = cv2.split(img)
            return cv2.merge((b.T, g.T, r.T))
        elif angle % 4 == 2:
            return self.mirror(img, 2)
        elif angle % 4 == 3:
            return self.mirror(self.spin(img, 1), 1)
        else:
            return img

    # # 这一步是没有必要的 已被弃用
    def delete_similar_image(self):
        file_list = os.listdir(self.folder_path)
        for i in range(len(file_list)):
            for j in range(len(file_list)):
                if i != j:
                    ret, score = self.compare_similarity(os.path.join(self.folder_path, file_list[i]),
                                                         os.path.join(self.folder_path, file_list[j]))
                    if ret:
                        print(file_list[i], file_list[j], score)

    def delete_doc_not_strict(self, threshold=0.6):
        print("开始去掉图片库%s中的文档或无背景图" % self.folder_path)
        count = 0
        for img_path in os.listdir(self.folder_path):
            img_path = self.folder_path + img_path
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if img is not None:
                if img[img == 255].size / img.size > threshold:
                    count += 1
                    os.remove(img_path)
        print("共删除了%d张疑似文档的图片" % count)
        print()

    @staticmethod
    def compare_similarity(img1_path, img2_path, threshold=0.8):
        img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), -1)
        img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), -1)
        if img1.shape != img2.shape:
            return False, 0
        else:
            gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            score, diff = structural_similarity(gray_1, gray_2, full=True)
            if score > threshold:
                return True, score
            else:
                return False, score

    def resize(self):
        flags = ["\\", "|", "/", "|"]
        for i, file_name in enumerate(os.listdir(self.folder_path)):
            full_name = self.folder_path + file_name
            img = cv2.imdecode(np.fromfile(full_name, dtype=np.uint8), -1)
            save_name = "%s%s_%d.png" % (self.folder_path, self.class_name, i)
            save_file = cv2.resize(img, (400, 400))
            cv2.imencode('.png', save_file)[1].tofile(save_name)
            if save_name != full_name:
                os.remove(full_name)
            print("\r", flags[i % 4], "已格式化%d张图片, 保存为:%s_%d.png" % (i + 1, self.class_name, i), end='')
        print()

    def set_folder(self, train_folder=TRAIN_SET_PATH, test_folder=TEST_SET_PATH, ratio=3):
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)
        dist = [test_folder]
        for _ in range(ratio):
            dist.append(train_folder)
        flags = ["\\", "|", "/", "|"]
        for i, file_name in enumerate(os.listdir(self.folder_path)):
            full_name = self.folder_path + file_name
            new_name = dist[i % (ratio + 1)] + "/" + file_name
            shutil.copy(full_name, new_name)
            print("\r", flags[i % 4],
                  "已移动%d张图片, 本次移动%s从%s到%s" % (i + 1, file_name, self.folder_path, dist[i % (ratio + 1)]), end='')
        print()

    @staticmethod
    def main_resize(img, _zero_flag=False, bi_threshold=127, _threshold=400 * 200):
        # # 二值变换 提高准确度
        if img is None:
            return
        _, bi_value = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), bi_threshold, 255, cv2.THRESH_BINARY)
        # # 获取图像轮廓
        contours, _ = cv2.findContours(bi_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # # 获取最大的轮廓
        _max_cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
        # # 求轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(_max_cnt)
        if w * h > _threshold:
            if _zero_flag:
                img[0: y, :] = 0
                img[y + h: -1, :] = 0
                img[:, 0: x] = 0
                img[:, x + w: -1] = 0
                return img, (x, y, w, h)
            else:
                return img[y: y + h, x: x + w], (x, y, w, h)
        else:
            return img, (x, y, w, h)


class Merge:
    def __init__(self):
        pass

    @staticmethod
    def get_class_label(file_path='蟹壳_24.png'):
        _pos = file_path.find("_")
        if _pos != -1:
            label = file_path[:_pos]
            return label, BY_LABEL[label]
        else:
            return None, -1

    @staticmethod
    def delete_invalid_files(image_root):
        for name in os.listdir(image_root):
            full_name = image_root + "/" + name
            img = cv2.imdecode(np.fromfile(full_name, np.uint8), -1)
            if len(img.shape) != 3 or img.shape[2] != 3:
                print("已删除", name, img.shape)
                os.remove(full_name)

    def merge(self, image_root='test_merge', step=500, set_label=''):
        self.delete_invalid_files(image_root)
        file_list = os.listdir(image_root)
        shuffle(file_list)
        count = 0
        for i in range(len(file_list) // step + 1):
            full_name = image_root + "/" + file_list[i * step]
            temp_image = cv2.imdecode(np.fromfile(full_name, np.uint8), -1)
            temp_label = [self.get_class_label(file_list[i * step])[1]]
            img = None
            for file_path in file_list[i * step + 1: (i + 1) * step]:
                try:
                    full_name = image_root + "/" + file_path
                    img = cv2.imdecode(np.fromfile(full_name, np.uint8), -1)
                    temp_image = np.hstack((temp_image, img))
                    temp_label.append(self.get_class_label(file_path)[1])
                    print("\r已处理%5d张图片, %20s" % (count, file_path), end='')
                    count += 1
                except Exception as e:
                    print(img.shape, file_path)
                    raise e
            print()
            cv2.imencode('.png', temp_image)[1].tofile(set_label + "_images_%d.png" % i)
            np.save(set_label + '_labels_%d.npy' % i, np.array(temp_label))
            print(len(temp_label), temp_image.shape)


if __name__ == "__main__":
    ret_images = []
    ret_labels = []
    image = cv2.imdecode(np.fromfile(TRAIN_SET_PATH + "/train_images_1.png", dtype=np.uint8), -1)
    ret_images.extend(np.hsplit(image, image.shape[1] // 400))
    ret_labels.extend(list(np.load(TRAIN_SET_PATH + "/train_labels_1.npy")))
    for i, image in enumerate(ret_images):
        ret = PreTreatment.main_resize(image)[0]
        print(BY_CLASS[ret_labels[i]])
        cv_show(ret)
    # Merge().merge(TEST_SET_PATH, set_label='test')
    # Merge().merge(TRAIN_SET_PATH, set_label=SRC_ROOT + '/new_train_set/train')
    # file_path = r"C:/Users/QQ863/Documents/Projects/PycharmProjects/GarbageClassify/src/test_set/易拉罐_87.png"
    # img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    # if img is not None:
    #     ret = PreTreatment.histogram_equalization(img)
    #     print(ret)
    # for key, value in IMAGE_LABELS.items():
    #     if not os.path.exists(os.path.join(IMAGES_ROOT, key)):
    #         os.mkdir(os.path.join(IMAGES_ROOT, key))
    #     for class_name in value:
    #         baidu_spider(class_name, key)
    #         folder = IMAGES_ROOT + "/" + key + "/" + class_name
    #         p = PreTreatment(folder, _class_name=class_name, need_check=True)
    #         p.resize()
    #         p.set_folder()
    #         print("已完成,", folder)
    # p = PreTreatment("test_set", _class_name='', need_check=False)
    # p.delete_doc_strict()
    # p = PreTreatment("train_set", _class_name='', need_check=False)
    # p.delete_doc_strict()
