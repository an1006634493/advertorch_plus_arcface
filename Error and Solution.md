1. 不知名报错：
```
Traceback (most recent call last):
  File "attacks1.py", line 85, in <module>
    learner.model, loss_fn=attack_cosine_distance(target=target_test), eps=3,
  File "attacks1.py", line 20, in __init__
    margin, size_average, reduce, reduction)
TypeError: __init__() takes from 1 to 4 positional arguments but 5 were given
```

解决：直接给`nn.CosineEmbeddingLoss`传空参：
```
class attack_cosine_distance(nn.CosineEmbeddingLoss):
    def __init__(self, target, margin=0., size_average=None, reduce=None, reduction='mean'):
        super(attack_cosine_distance, self).__init__()
        self.target = target
```
