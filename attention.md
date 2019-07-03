不知名报错：
Traceback (most recent call last):
  File "attacks1.py", line 85, in <module>
    learner.model, loss_fn=attack_cosine_distance(target=target_test), eps=3,
  File "attacks1.py", line 20, in __init__
    margin, size_average, reduce, reduction)
TypeError: __init__() takes from 1 to 4 positional arguments but 5 were given

解决：直接给nn.CosineEmbeddingLoss传空参：
class attack_cosine_distance(nn.CosineEmbeddingLoss):
    def __init__(self, target, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(attack_cosine_distance, self).__init__()
        self.target = target

攻击存在的问题：

可以对模型进行攻击，但是原方法选用的损失函数将无法使用。（如Arcface、Cosface等）
原因：detection的工作在训练时进行的是类似classification的过程，loss函数针对最后一个FC层设计，但

在进行detection时，只进行到feature embedding，然后计算欧氏距离。所以无法应用最后一层之后的loss 

function。

若直接攻击，有两种方法：
1、直接计算特征间的相似度作为loss函数；
2、强行用FC进行分类（即使用训练集的类别进行分类），然后攻击到训练集中另一类别去。感觉不如第一种

方法。

问题：
怎么做到识别出unknown的？
检测人脸和识别人脸分开进行，检测到的人脸若无匹配，则归为unknown。


对classification和对detection的攻击有区别。现阶段大部分工作都是对的分类的攻击。对detection的攻击

也都是针对训练集和测试集类别相同的模型。

能不能直接针对训练集攻击？

注意：
1、别用conf.ce_loss，直接用nn.CrossEntropyLoss()。
2、 在conf的infer下也加入conf.pin_memory = True和conf.num_workers = 3，在计算class_num时用到。
3、复制文件夹使用cp时，若目录下还有目录，直接cp会失败，加上-r。
4、对tqdm，要from tqdm import tqdm，否则会有报错：TypeError: 'module' object is not callable
5、把conf里的batch_size改小，要不然显存不够
6、注释里面有中文也会报错：
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb8 in position 3832: invalid start byte
7、若a是tensor，查找a中最大元素不能用a.index(max(a))，因为index函数只对list有用，所以要用：
list(a).index(max(a))
