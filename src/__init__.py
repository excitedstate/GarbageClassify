"""
1. 先对图片进行一次相似性去重
2. 对部分图片进行一次数据增强
3. 然后对图片进行分区
4. 选出'最好'的几个区
5. 提取特征区特征 (或重组后作为输入 长宽最好是2的幂)
## OK
6. 输入到CNN网络
7. 训练 / 预测
8. 与其他机器学习方法的比较
"""