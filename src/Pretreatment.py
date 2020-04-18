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

# # # 图像分割算法
class PicSplit:
    def __init__(self):
        self.img = cv2.imread(TMP_ROOT + "/test.jpg")
        # cv_show(self.img, "origin-img")

    def watershed(self, _img=None):
        # # 灰度和二值转换
        _img = self.img if _img is None else _img
        _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _, _binary = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # # 形态学操作
        # # # 形态学操作卷积核
        _kernel = np.ones((3, 3), np.uint8)
        # # # 开运算去噪(去掉椒盐噪声的影响)
        _opening = cv2.morphologyEx(_binary, cv2.MORPH_OPEN, _kernel, iterations=2)
        # # # 如果能画出背景和前景, 分割算法会很好
        # # # 考虑到数据量的原因, 使用程序 机械的找出
        # # # 找出一定是背景的部分 膨胀操作: 扩大图形区的面积
        _sure_bg = cv2.dilate(_opening, _kernel, iterations=3)
        # cv_show(_sure_bg)
        # # 距离变换函数: 对原始图像进行计算 之后二值处理, 获取前景
        # # 该函数的第一个参数只能是单通道的二值的图像, 第二个参数是距离方法
        # # 计算图像上255点与最近的0点之间的距离 DIST_L2应是欧氏距离, 会输出小数
        # # DIST_L1应是哈密顿距离, 不会有小数
        _dist_transform = cv2.distanceTransform(_opening, cv2.DIST_L1, 5)
        # cv_show(_dist_transform)
        # # 距离变换之后做一二值变换, 得到大概率是图像前景的点
        _, _sure_fg = cv2.threshold(_dist_transform, 0.5 * _dist_transform.max(), 255, cv2.THRESH_BINARY)
        # # 转换类型, 否则会很危险
        _sure_fg = np.uint8(_sure_fg)
        # cv_show(_sure_fg)
        # # 绘制unknown区 交给算法, 自下而上的洪泛算法
        _unknown = cv2.subtract(_sure_bg, _sure_fg)
        # cv_show(_unknown)
        _, _markers = cv2.connectedComponents(_sure_fg)
        _markers = _markers + 1
        _markers[_unknown == 255] = 0
        _img1 = _img.copy()
        _markers = cv2.watershed(_img1, _markers)

        # # 圈出来 之后可以根据结果将一部分的值变为黑色
        def random_color(a: int):
            return np.random.randint(0, 255, (a, 3))

        _markers_label = np.unique(_markers)
        _colors = random_color(_markers_label.size)
        for _mark, _color in zip(_markers_label, _colors):
            _img1[_markers == _mark] = _color
        # # 展示
        cv_show(_img1)

    def to_counters(self, _img=None):
        # # 检测轮廓 边缘分割
        _img = _img if _img is not None else self.img
        # # 灰度和二值
        _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _, _binary = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv_show(_binary, "binary test")
        # # Canny边缘检测
        # edges = cv2.Canny(_img, 50, 150, )
        # cv_show(edges)
        # # 轮廓检测, 画出轮廓
        _contours, _ = cv2.findContours(_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _draw_img = _img.copy()
        # # 计算轮廓的大小, 选定阈值 对轮廓进行选择
        _greatest_contour_idx = np.argmax([cv2.contourArea(_contour) for _contour in _contours])
        # # 求轮廓的外接矩形
        _greatest_contour = _contours[_greatest_contour_idx]
        _x, _y, _w, _h = cv2.boundingRect(_greatest_contour)
        _ret = cv2.rectangle(_img, (_x, _y), (_x + _w, _y + _h), color=(0, 255, 255), thickness=1)
        # # 上下
        _ret[0: _y, :] = (0, 0, 0)
        _ret[_y + _h: -1, :] = (0, 0, 0)
        # # 左右
        _ret[:, 0: _x] = (0, 0, 0)
        _ret[:, _x + _w: -1] = (0, 0, 0)
        # _ret = cv2.drawContours(_draw_img, _contours, _greatest_contour_idx, (255, 0, 255), 2)
        cv_show(_ret)

    """
    其他图像分割方法 还有机器学习方法 时间关系 不学了
     1. 深度学习图像分割  labelme & FCN --> 将这个分割方法作为重点学习
     2. 基于区域的分割
        (1) 区域生长算法
            a. 选择合适的生长点(seed)
            b. 确定相似性准则即生长规则(生长点像素和当前像素之差小于阈值T, 自定义的或者是其他算法给出的, 如遗传算法)
            c. 确定生长停止条件
            d. 从seed点附近开始按照相似性准则拓展, 直到不能拓展(图像边界 或不适合遇到一圈不合适的点)
            e. 遍历整张图像, 执行a~e, 直到图像中没有可以未被检查的点
        (2) 区域分离和聚合算法
            a. 分割子区域(sub_zone)(4个子区域, 不断分割, 直到不能再分, 要基于自己设定的条件)
            b. 确定合并规则(若B区中有80%的点满足以下关系: A区均值与B区某点的像素值差的绝对值<=2(A区方差))
            c. 自叶子结点开始向上遍历子区域, 确定能否合并
                c1. 能合并, 合并之后作为新的叶子结点
                c2. 不能合并, continue
            d. 根节点与叶子结点直接相连, 递归过程结束
    """

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
