import numpy as np
import pyrr
from OpenGL.GL import *
import ctypes
from math import radians

class GameObject:
    def __init__(self, vertices, indices, position, color):
        self.position = np.array(position, dtype=np.float32)
        self.vertices = vertices
        self.indices = indices
        self.color = color
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]
        self.setup_mesh()

    def setup_mesh(self):
        data = []
        for i in range(0, len(self.vertices), 3):
            data.extend(self.vertices[i:i+3])
            data.extend(self.color)
            data.extend([0.0, 0.0])

        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(data, dtype=np.float32).nbytes,
                     np.array(data, dtype=np.float32), GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes,
                     self.indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

    def get_model_matrix(self):
        model = pyrr.matrix44.create_identity()
        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_translation(self.position)
        )
        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_scale(self.scale)
        )
        return model

    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)