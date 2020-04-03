# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #

IMAGE_LABELS = {
    "湿垃圾": [
        "菜叶", "橙皮", "葱", "饼干", "番茄酱",
        "蛋壳", "西瓜皮", "马铃薯", "鱼骨", "甘蔗",
        "玉米", "鸡骨", "虾壳", "蛋糕", "面包",
        "草莓", "西红柿", "梨", "蟹壳", "香蕉皮",
        "辣椒", "巧克力", "茄子", "豌豆皮", "苹果",
        "树叶", "花生壳"
    ],
    "干垃圾": [
        "贝壳", "化妆刷", "海绵", "发胶", "卫生纸",
        "旧镜子", "核桃", "陶瓷碗", "一次性筷子", "西梅核",
        "坏的花盆", "脏污衣服", "烟蒂", "湿垃圾袋", "扫把",
        "牙刷", "过期化妆品", "牙膏皮", "水彩笔", "调色板",
        "打火机", "荧光棒", "医用手套", "医用纱布", "医用棉签", "创可贴",
        "注射器"
    ],
    "可回收垃圾": [
        "塑料瓶", "食品罐头", "玻璃瓶", "易拉罐", "报纸",
        "旧书包", "旧手提包", "旧鞋子", "牛奶盒", "旧塑料篮子",
        "旧玩偶", "玻璃壶", "旧铁锅", "垃圾桶", "塑料梳子",
        "旧帽子", "旧夹子", "废锁头", "篮球", "旧纸袋",
        "纸盒", "旧玩具", "木梳子", "香水瓶", "煤气罐",
    ],
    "有害垃圾": [
        "油漆桶", "电池", "油漆",
        "过期的胶囊药物", "含汞温度计", "过期药片",
        "荧光灯", "蓄电池", "杀虫剂"
    ]
}

PROJECT_ROOT = r"../GarbageClassify"
IMAGES_ROOT = PROJECT_ROOT + r"/image"
LOGS_ROOT = PROJECT_ROOT + r"/log"
SRC_ROOT = PROJECT_ROOT + r"/src"
TMP_ROOT = PROJECT_ROOT + r"/tmp"
TRAIN_SET_PATH = SRC_ROOT + r"/train_set"
TEST_SET_PATH = SRC_ROOT + r"/test_set"

# # 以下配置信息暂不启用
MySQL_CONFIG = {
    "host": "set your host",
    "port": 3306,
    "user": "set your username",
    "password": "set your password",
    "database": "set your database"
}
BY_CLASS = {0: '菜叶', 1: '橙皮', 2: '葱', 3: '饼干', 4: '番茄酱', 5: '蛋壳', 6: '西瓜皮', 7: '马铃薯', 8: '鱼骨', 9: '甘蔗', 10: '玉米',
            11: '鸡骨', 12: '虾壳', 13: '蛋糕', 14: '面包', 15: '草莓', 16: '西红柿', 17: '梨', 18: '蟹壳', 19: '香蕉皮', 20: '辣椒',
            21: '巧克力', 22: '茄子', 23: '豌豆皮', 24: '苹果', 25: '树叶', 26: '花生壳', 27: '贝壳', 28: '化妆刷', 29: '海绵', 30: '发胶',
            31: '卫生纸', 32: '旧镜子', 33: '核桃', 34: '陶瓷碗', 35: '一次性筷子', 36: '西梅核', 37: '坏的花盆', 38: '脏污衣服', 39: '烟蒂',
            40: '湿垃圾袋', 41: '扫把', 42: '牙刷', 43: '过期化妆品', 44: '牙膏皮', 45: '水彩笔', 46: '调色板', 47: '打火机', 48: '荧光棒',
            49: '医用手套', 50: '医用纱布', 51: '医用棉签', 52: '创可贴', 53: '注射器', 54: '塑料瓶', 55: '食品罐头', 56: '玻璃瓶', 57: '易拉罐',
            58: '报纸', 59: '旧书包', 60: '旧手提包', 61: '旧鞋子', 62: '牛奶盒', 63: '旧塑料篮子', 64: '旧玩偶', 65: '玻璃壶', 66: '旧铁锅',
            67: '垃圾桶', 68: '塑料梳子', 69: '旧帽子', 70: '旧夹子', 71: '废锁头', 72: '篮球', 73: '旧纸袋', 74: '纸盒', 75: '旧玩具', 76: '木梳子',
            77: '香水瓶', 78: '煤气罐', 79: '油漆桶', 80: '电池', 81: '油漆', 82: '过期的胶囊药物', 83: '含汞温度计', 84: '过期药片', 85: '荧光灯',
            86: '蓄电池', 87: '杀虫剂'}
BY_LABEL = dict(zip(BY_CLASS.values(), BY_CLASS.keys()))
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #
