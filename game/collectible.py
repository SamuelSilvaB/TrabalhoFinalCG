from enum import Enum
import numpy as np

class CollectibleType(Enum):
    BUFF = 1
    NERF = 2

class Collectible:
    def __init__(self, render_object, ctype):
        """
        render_object: Objeto que será renderizado (StarObject, SphereObject, etc)
        ctype: CollectibleType.BUFF ou CollectibleType.NERF
        """
        self.render_object = render_object
        self.type = ctype
        self.position = render_object.position
    
    def update(self, speed):
        self.position[2] += speed
        self.render_object.position[2] = self.position[2]
        
        # Rotação animada
        self.render_object.rotation[1] += 3.0  # Gira no eixo Y
        self.render_object.rotation[2] += 1.5  # Gira no eixo Z (opcional)
    
    def get_model_matrix(self):
        return self.render_object.get_model_matrix()
    
    def draw(self):
        self.render_object.draw()