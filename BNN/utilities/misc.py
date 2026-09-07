import torch


# %%
class NestedTensor:
    def __init__(self, left, right, disp=None, ref=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.ref = ref


def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(source.shape))
    ]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)
