# %%


class NestedTensor:
    def __init__(self, left, right, disp=None, ref=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.ref = ref
