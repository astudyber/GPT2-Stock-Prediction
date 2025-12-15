import os
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常
plt.figure(figsize=(20, 20))  # 设置图像的宽和高


class MyVal:

    # 初始化，加载数据路径，是否保存json结果
    def __init__(self, path1 = './val/', path2='./val_model/', save_json=False, class_name=None, iou_threshold=0.5):
        '''
        1: 真实标签
        2: 预测标签
        '''
        self.p_class = None
        self.r_class = None
        self.p_all = None
        self.r_all = None
        self.AP_class = []
        if class_name is None:
            class_name = ['car', 'truck ', 'bus ', 'person', 'fire', 'smoke', ' cone', 'div', 'suit', 'box', 'moto']
        self.path1 = path1 # 正确标签路径 path1
        self.path2 = path2 # 预测标签路径 path2
        self.save_json = save_json # 是否将验证结果保存为json
        self.class_name = class_name

        # 获取 val 和 val_model 所有的txt文件的名称【files_name1】和【files_name2】
        with os.scandir(self.path1) as entries:
            self.files_name1 = [entry.name for entry in entries if entry.is_file()]
        with os.scandir(self.path1) as entries:
            self.files_name2 = [entry.name for entry in entries if entry.is_file()]

        # 预测情况的数组，每个元素为（图片名；类别class；置信度confidence；是否正确预测result）
        self.forecast_situation = []
        self.forecast_situation_json = {}
        # 真实的标签总数（召回率R的分母）
        self.real_label_num = 0
        self.real_label_num_child = [0 for _ in range(len(class_name))]
        # 调用函数初始化上面的3个数值
        self.statistics(iou_threshold = iou_threshold)

        # self.forecast_situation 的分类与排序：得到self.forecast_situation_child
        self.forecast_situation_child = [[] for _ in range(len(class_name))] # [[第0类预测结果按置信度排序], [第1类预测结果按置信度排序], ... ]
        self.classification()

        # 绘制 PR 曲线图像
        self.PR()

        # 计算最终的 map
        self.map()


    # 计算  两个标签之间的  iou--------阈值直接决定了模型的评价指标的优劣
    # 输入：真实标签，预测标签，iou阈值
    # 输出：True(同一物体)，False(预测失败，不是同一物体)
    def iou(self, label1, label2, iou_threshold=0.5):
        # 重叠部分的宽高
        width = (label1[3] + label2[3]) / 2 - abs(label1[1] - label2[1])
        hight = (label1[4] + label2[4]) / 2 - abs(label1[2] - label2[2])
        # 如果两个锚框有重叠
        if width>0 and hight>0 :
            iou = (width*hight) / (label1[3]*label1[4] + label2[3]*label2[4] - width*hight)
            print('\033[92m iou:{} \033[0m '.format(iou))
            # 高于阈值就返回 True
            if iou > iou_threshold:
                return True
            else:
                return False
        # 两个锚框没有重叠
        else:
            return False


    # 统计预测情况 self.forecast_situation（图片名；类别；置信度；是否正确预测）
    # 顺带统计了 self.real_label_num （召回率 R 的分母）；包括每一类
    # 判断所有的标签是否正确预测，保存到变量self.forecast_situation中；同时更新到新的文件 val_iou 和 val_iou_sli 中
    def statistics(self, iou_threshold=0.5):
        # 遍历真实标签
        for i in range(len(self.files_name1)):
            real = []  # 真实标签数组
            pred = []  # 预测标签数组

            # 读文件标签获取信息
            with open(self.path1 + self.files_name1[i], 'r') as f:
                # print('\033[92m 真实标签:  \033[0m ')
                for line in f.readlines():
                    line = line.strip().split()
                    line = [float(i) for i in line]
                    line[0] = int(line[0])
                    real.append(line)
                    self.real_label_num += 1 # 更新'分母'
                    self.real_label_num_child[line[0]] += 1
                    # print(line)

            # 判断是否存在预测文件（预测文件少一些嘛）
            file_path = self.path2 + self.files_name1[i]
            if os.path.exists(file_path):
                if self.save_json:
                    self.forecast_situation_json[self.files_name1[i]] = []
                with open(file_path, 'r') as f:
                    # print('\033[92m 预测标签:  \033[0m ')
                    for line in f.readlines():
                        line = line.strip().split()
                        line = [float(i) for i in line]
                        line[0] = int(line[0])
                        pred.append(line)
                        # print(line)

                # 判断一个照片的预测标签 pred 是否正确
                for j, pre in enumerate(pred):
                    for rea in real:
                        # 如果预测正确
                        if pre[0]==rea[0] and self.iou(rea, pre, iou_threshold):
                            pred[j].append(1)

                    # 预测正确
                    if pred[j][-1] == 1:
                        print('\033[92m 预测++正确  \033[0m ')
                        self.forecast_situation.append([self.files_name1[i], pre[0], pre[5], 1])  # （图片名；类别class；置信度confidence；是否正确预测result）
                        if self.save_json:
                            self.forecast_situation_json[self.files_name1[i]].append({
                                'class': pre[0],
                                'confidence': pre[5],
                                'result': 1
                            })
                    # 预测错误
                    else:
                        print('\033[91m 预测--错误  \033[0m ')
                        pred[j].append(0)
                        self.forecast_situation.append([self.files_name1[i], pre[0], pre[5], 0])
                        if self.save_json:
                            self.forecast_situation_json[self.files_name1[i]].append({
                                'class': pre[0],
                                'confidence': pre[5],
                                'result': 1
                            })

                # 将判断结果写入JSON文件
                if self.save_json:
                    with open('./val_iou.json', 'w', encoding='utf-8') as file:
                        json.dump(self.forecast_situation_json, file, ensure_ascii=False, indent=4)


    # 分类与排序
    # 根据类别生成置信度排序的
    def classification(self):
        # 按照置信度进行排序：使用sorted函数对列表进行排序，置信度位于子数组的第三个位置，即索引2
        self.forecast_situation = sorted(self.forecast_situation, key=lambda x: x[2], reverse=True)

        # 进行类别划分
        for i in range(len(self.forecast_situation)):
            for j in range(len(self.class_name)):
                if j == self.forecast_situation[i][1]:
                    self.forecast_situation_child[j].append(self.forecast_situation[i])
                    break


    # 计算指标
    # （预测正确T；预测错误F）（预测为正例P；预测为反例N）
    #
    # 分母：在实际yolo标注的标签中，显然只有真实的正例（TP 和 FN），因此 R 的分母就是所有真实标签的数量；同样的，在预测结果中，显然只有预测的正例（TP 和 FP），因此 P 的分母就是当前阈值下预测标签的数量
    # 分子均为：特定阈值下，预测正确的标签数量
    #
    # 准确率 P（预测的所有正样本中，预测正确的比例） P = TP /（TP + FP） ：特定阈值下，分母为置信度超过阈值的预测的标签数量
    # 召回率 R（实际的所有正样本中，预测正确的比例） R = TP /（TP + FN） ：任何阈值下，分母均为真实的标签数量                           self.real_label_num （召回率 R 的分母）
    # PR曲线：按照每个样本的置信度概率将它们从大到小排序，不同的置信度阈值得到多个样本集，分别计算出多组P、R值，从而绘制出P曲线、R曲线、PR曲线
    #
    # 注意：每加入一个样本，不是TP+1就是FP+1；P始终改变，R可能不变（出现了FP+1）；此时同一个R对应多个P，仅保留max(P);同时注意到出现FP+1时，一定导致P下降，取首次P值即为max(P)，也就是无需更新P和R
    def PR(self, show=False):
        # 首先是总的 P R 计算
        self.r_all = []
        self.p_all = []
        TP = 0
        for i, c in enumerate(self.forecast_situation):  # 容易注意到 i+1 就是 P 的分母
            # 如果是 TP+1 导致PR均改变
            if c[3] == 1:
                TP += 1
                self.r_all.append(TP/self.real_label_num)
                self.p_all.append(TP/(i+1))

        # 然后按照类别 P R 计算
        self.r_class = [[] for _ in range(len(self.class_name))]
        self.p_class = [[] for _ in range(len(self.class_name))]
        for j in range(len(self.class_name)):
            TP = 0
            for i, c in enumerate(self.forecast_situation_child[j]):  # 容易注意到 i+1 就是 P 的分母
                # 如果是 TP+1 导致PR均改变
                if c[3] == 1:
                    TP += 1
                    self.r_class[j].append(TP / self.real_label_num_child[j])
                    self.p_class[j].append(TP / (i + 1))
        # for it in self.p_class:
        #     print(it)

        # 画图
        fig, ax = plt.subplots()
        ax.plot(self.r_all, self.p_all, linewidth=0.1, marker='o', markersize=0.1, markeredgecolor='blue')  #折线图
        for index in range(len(self.class_name)):
            ax.plot(self.r_class[index], self.p_class[index], linewidth=0.3, marker='o', markersize=0.1)  #折线图

        ax.set_xlabel('召回率 R')  # 添加标题和坐标轴标签
        ax.set_ylabel('准确率 P')
        ax.set_title('PR曲线')
        plt.xlim(0, 1)  # 设置横坐标的范围为0到1
        plt.ylim(0, 1)
        plt.xticks([i / 10 for i in range(11)])  # 设置纵坐标的刻度
        plt.yticks([i/10 for i in range(11)])
        plt.savefig('PR.png', dpi=1000)  # 保存图形到本地
        if show:
            plt.show()  # 显示图像


    # 计算指标
    # AP：平均精度（某一类的），某一类别中PR曲线的面积【P的均值】
    # mAP：平均均值精度（整体的），对所有类别的AP求均值
    def map(self):
        # 防止分母为空，记录哪些值为空
        lab = []
        # 计算AP
        for j, s in enumerate(self.p_class):
            if len(s) == 0:
                lab.append(j)
            else:
                self.AP_class.append(round(sum(s)/len(s), 5))
        # 非空的类别数组
        self.class_name_true = [name for k, name in enumerate(self.class_name) if k not in lab]
        for i in range(len(self.class_name_true)):
            print('\033[93m {} '.format(self.class_name_true[i])+'\t'*int(5-len(self.class_name_true[i])/4+0.25) +' AP:{} \033[0m '.format(self.AP_class[i]))

        # 计算mAP
        self.mAP = round(sum(self.AP_class)/len(self.AP_class), 4)
        print('\033[94m\n all \t\t\t\t mAP:{} \033[0m '.format(self.mAP))




if __name__ == '__main__':
    # 基本类别划分
    # class_name = ['car', 'truck ', 'bus ', 'person', 'fire', 'smoke', 'cone', 'div', 'suit', 'box', 'moto']
    class_name = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']

    # 验证
    # val = MyVal(path1 = './val/', path2='./val_model/', save_json=False, class_name=class_name, iou_threshold=0.5)
    val = MyVal(path1 = './val/', path2='./val_model_sli/', save_json=False, class_name=class_name, iou_threshold=0.5)

    # val.PR(True)  # 显示PR曲线图像


