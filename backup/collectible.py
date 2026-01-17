from enum import Enum
from game_object import GameObject  # ajuste o import

class CollectibleType(Enum):
    BUFF = 1
    NERF = 2

class Collectible(GameObject):
    def __init__(self, vertices, indices, position, color, ctype):
        super().__init__(vertices, indices, position, color)
        self.type = ctype