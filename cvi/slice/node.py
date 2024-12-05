class CviNode:
    loc: float
    crv: float

    def __init__(self, loc: float, crv: float):
        self.loc = loc
        self.crv = crv

    def is_zero(self):
        return self.loc == 0.0
