import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from math import sin, cos, radians
import random
from PIL import Image



def load_shader_from_file(path):
    with open(path, 'r') as file:
        return file.read()


class WaterPlane3D:
    def __init__(self, size=200, tile=100):
        s = 200

        vertices = np.array([       # Cor       #UV         # Normal
            -size, -0.5, -s,        1,1,1,   0,    0,         0,1,0,  
             s, -0.5, -s,           1,1,1,   tile, 0,         0,1,0,
             s, -0.5,  s,           1,1,1,   tile, tile,      0,1,0,
            -s, -0.5,  s,           1,1,1,   0,    tile,      0,1,0
        ], dtype=np.float32)

        indices = np.array([0,1,2, 0,2,3], dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        stride = 11 * 4  # 11 floats
        # posição
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # cor
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        # uv
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        # NORMAL (NOVO)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))


        self.texture = load_texture("texture/water.jpg")

    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


def load_texture(path):
    """Carrega uma textura de imagem"""
    try:
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.convert("RGB").tobytes()
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        print(f"✓ Textura carregada: {path}")
        return texture
    except Exception as e:
        print(f"✗ Erro ao carregar textura {path}: {e}")
        return None

class GameObject:
    def __init__(self, vertices, indices, position, color):
        self.position = np.array(position, dtype=np.float32)
        self.vertices = vertices
        self.indices = indices
        self.color = color
        self.rotation = [0, 0, 0]
        self.scale = [1, 1, 1]
        self.texture = None
        self.setup_mesh()
    
    def setup_mesh(self):
        colored_vertices = []
        for i in range(0, len(self.vertices), 3):
            colored_vertices.extend([self.vertices[i], self.vertices[i+1], self.vertices[i+2]])
            colored_vertices.extend(self.color)
            colored_vertices.extend([0.0, 0.0])  # UV padrão
        
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)
        
        glBindVertexArray(self.VAO)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(colored_vertices, dtype=np.float32).nbytes,
                     np.array(colored_vertices, dtype=np.float32), GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        # Posição
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        
        # Cor
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        
        # UV
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    
    def get_model_matrix(self):
        model = pyrr.matrix44.create_identity()
        
        model = pyrr.matrix44.multiply(model, 
            pyrr.matrix44.create_from_translation(self.position))
        
        if self.rotation[1] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_y_rotation(radians(self.rotation[1])))
        if self.rotation[0] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_x_rotation(radians(self.rotation[0])))
        if self.rotation[2] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_z_rotation(radians(self.rotation[2])))
        
        model = pyrr.matrix44.multiply(model,
            pyrr.matrix44.create_from_scale(self.scale))
        
        return model
    
    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

class ModelObject:
    """Classe para carregar modelos .obj com múltiplas texturas"""
    def __init__(self, obj_path, position, color=[0.7, 0.7, 0.7], scale=1.0, texture_paths=None):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = [0, 0, 0]
        self.scale = [scale, scale, scale]
        self.color = color
        self.textures = []  # Lista de texturas
        self.materials = []  # Lista de materiais com suas geometrias
        
        # Carregar texturas se fornecidas
        if texture_paths:
            if isinstance(texture_paths, str):
                # Uma única textura
                tex = load_texture(texture_paths)
                if tex:
                    self.textures.append(tex)
            elif isinstance(texture_paths, list):
                # Múltiplas texturas
                for tex_path in texture_paths:
                    tex = load_texture(tex_path)
                    if tex:
                        self.textures.append(tex)
        
        # Carregar modelo .obj
        self.load_obj_with_materials(obj_path)
    
    def load_obj_with_materials(self, filename):
        """Carrega arquivo .obj com múltiplos materiais, UVs e NORMAIS (Phong-ready)"""

        vertices = []
        tex_coords = []
        normals = []

        current_material = "default"
        material_faces = {}

        print(f"Carregando modelo: {filename}")

        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()

                    # --------------------
                    # POSIÇÃO
                    # --------------------
                    if parts[0] == 'v':
                        vertices.append([
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3])
                        ])

                    # --------------------
                    # UV
                    # --------------------
                    elif parts[0] == 'vt':
                        tex_coords.append([
                            float(parts[1]),
                            float(parts[2]) if len(parts) > 2 else 0.0
                        ])

                    # --------------------
                    # NORMAL
                    # --------------------
                    elif parts[0] == 'vn':
                        normals.append([
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3])
                        ])

                    # --------------------
                    # MATERIAL
                    # --------------------
                    elif parts[0] == 'usemtl':
                        current_material = parts[1]
                        if current_material not in material_faces:
                            material_faces[current_material] = []

                    # --------------------
                    # FACE
                    # --------------------
                    elif parts[0] == 'f':
                        face = []

                        for i in range(1, len(parts)):
                            indices = parts[i].split('/')

                            v_idx = int(indices[0]) - 1
                            vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else 0
                            vn_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0

                            face.append((v_idx, vt_idx, vn_idx))

                        # triangulação
                        for i in range(1, len(face) - 1):
                            material_faces[current_material].append(
                                [face[0], face[i], face[i + 1]]
                            )

            print(f"✓ Vértices: {len(vertices)}, UVs: {len(tex_coords)}, Normais: {len(normals)}")
            print(f"✓ Materiais: {list(material_faces.keys())}")

            # --------------------
            # NORMALIZA MODELO
            # --------------------
            if vertices:
                v_np = np.array(vertices)
                center = (v_np.min(axis=0) + v_np.max(axis=0)) / 2
                size = v_np.max(axis=0) - v_np.min(axis=0)
                scale = max(size)

                for i in range(len(vertices)):
                    vertices[i] = (np.array(vertices[i]) - center) / scale

            if not tex_coords:
                tex_coords = [[0.0, 0.0]]

            # --------------------
            # CRIA VAO POR MATERIAL
            # --------------------
            for mat_idx, (mat_name, faces) in enumerate(material_faces.items()):
                vertices_array = []
                indices_array = []

                for face in faces:
                    for v_idx, vt_idx, vn_idx in face:
                        # posição
                        vertices_array.extend(vertices[v_idx])

                        # cor (tint)
                        vertices_array.extend(self.color)

                        # uv
                        if vt_idx < len(tex_coords):
                            vertices_array.extend(tex_coords[vt_idx])
                        else:
                            vertices_array.extend([0.0, 0.0])

                        # normal
                        if vn_idx < len(normals):
                            vertices_array.extend(normals[vn_idx])
                        else:
                            vertices_array.extend([0.0, 1.0, 0.0])

                        indices_array.append(len(indices_array))

                vertices_np = np.array(vertices_array, dtype=np.float32)
                indices_np = np.array(indices_array, dtype=np.uint32)

                VAO = glGenVertexArrays(1)
                VBO = glGenBuffers(1)
                EBO = glGenBuffers(1)

                glBindVertexArray(VAO)

                glBindBuffer(GL_ARRAY_BUFFER, VBO)
                glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_np.nbytes, indices_np, GL_STATIC_DRAW)

                stride = 11 * 4  # 11 floats

                # posição
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

                # cor
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

                # uv
                glEnableVertexAttribArray(2)
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

                # normal
                glEnableVertexAttribArray(3)
                glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))

                texture_idx = min(mat_idx, len(self.textures) - 1) if self.textures else None

                self.materials.append({
                    'name': mat_name,
                    'VAO': VAO,
                    'indices_count': len(indices_np),
                    'texture_idx': texture_idx
                })

                print(f"  • Material '{mat_name}' OK ({len(indices_np)//3} triângulos)")

        except Exception as e:
            print("✗ Erro ao carregar OBJ:", e)
            import traceback
            traceback.print_exc()
            raise

    
    def get_model_matrix(self):
        model = pyrr.matrix44.create_identity()
        
        model = pyrr.matrix44.multiply(model, 
            pyrr.matrix44.create_from_translation(self.position))
        
        if self.rotation[1] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_y_rotation(radians(self.rotation[1])))
        if self.rotation[0] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_x_rotation(radians(self.rotation[0])))
        if self.rotation[2] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_z_rotation(radians(self.rotation[2])))
        
        model = pyrr.matrix44.multiply(model,
            pyrr.matrix44.create_from_scale(self.scale))
        
        return model
    
    def draw(self):
        """Desenha todos os materiais do modelo"""
        for material in self.materials:
            glBindVertexArray(material['VAO'])
            glDrawElements(GL_TRIANGLES, material['indices_count'], GL_UNSIGNED_INT, None)

# COLISÃO AABB
PLAYER_SIZE = np.array([0.8, 0.6, 1.2])   # largura, altura, profundidade
OBSTACLE_SIZE = np.array([0.8, 0.6, 1.2])

def aabb_collision(posA, sizeA, posB, sizeB):
    minA = posA - sizeA / 2
    maxA = posA + sizeA / 2

    minB = posB - sizeB / 2
    maxB = posB + sizeB / 2

    return (
        minA[0] <= maxB[0] and maxA[0] >= minB[0] and
        minA[1] <= maxB[1] and maxA[1] >= minB[1] and
        minA[2] <= maxB[2] and maxA[2] >= minB[2]
    )

# DEBUG COLISAO
def create_hitbox_vao():
    # Cubo unitário centrado na origem
    vertices = np.array([
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5,
    ], dtype=np.float32)

    indices = np.array([
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7
    ], dtype=np.uint32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    return VAO, len(indices)


class NauticRunner:
    def __init__(self):
        if not glfw.init():
            raise Exception("GLFW não pode ser inicializado!")
        
        self.width, self.height = 1280, 720
        self.window = glfw.create_window(self.width, self.height, "Nautic Runner", None, None)
        
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window não pode ser criada!")
        
        glfw.set_window_pos(self.window, 400, 200)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.make_context_current(self.window)
        
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.3, 0.5, 1.0)
        
        vertex_code = load_shader_from_file("shaders/vertex.glsl")
        fragment_code = load_shader_from_file("shaders/fragment.glsl")

        self.shader = compileProgram(
            compileShader(vertex_code, GL_VERTEX_SHADER),
            compileShader(fragment_code, GL_FRAGMENT_SHADER)
        )

        
        # Game state
        self.player_lane = 1
        self.player_y = 0.0
        self.jumping = False
        self.jump_velocity = 0.0
        self.game_speed = 0.1
        self.score = 0
        self.game_over = False
        self.using_custom_model = False

        self.water = WaterPlane3D()
        
        # Configuração das pistas
        self.lane_positions = [-1.0, 0.0, 1.0]
        
        self.setup_game_objects()
        
        # Projection matrix
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.width/self.height, 0.1, 100
        )
        
        # View matrix
        self.view = pyrr.matrix44.create_look_at(
            pyrr.Vector3([0, 3, 8]),
            pyrr.Vector3([0, 0, 0]),
            pyrr.Vector3([0, 1, 0])
        )

        self.hitbox_vao, self.hitbox_index_count = create_hitbox_vao()
        self.show_hitboxes = False

    def draw_hitbox(self, position, size, scale):
        model = pyrr.matrix44.create_identity()

        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_translation(position)
        )

        # tamanho da hitbox * escala do objeto
        final_scale = np.array(size) * np.array(scale)

        model = pyrr.matrix44.multiply(
            model,
            pyrr.matrix44.create_from_scale(final_scale)
        )

        model_loc = glGetUniformLocation(self.shader, "model")
        use_texture_loc = glGetUniformLocation(self.shader, "useTexture")

        glUniform1i(use_texture_loc, 0)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        glBindVertexArray(self.hitbox_vao)
        glDrawElements(GL_LINES, self.hitbox_index_count, GL_UNSIGNED_INT, None)
    


    
    def setup_game_objects(self):
        try:
            # CONFIGURE AQUI: caminho do modelo e LISTA de texturas
            model_path = "Boat/boat.obj"
            
            # OPÇÃO 1: Múltiplas texturas (uma para cada material)
            texture_paths = [
                "Boat/Texture/wood1.jpg",  # Material 1 
                "Boat/Texture/wood2.jpg",  # Material 2 
                "Boat/Texture/wood3.jpg",  # Material 3 
            ]
            
            # OPÇÃO 2: Uma única textura para todo o modelo
            # texture_paths = "Boat/boat_texture.png"
            
            self.player = ModelObject(model_path, 
                                     [self.lane_positions[1], 0, 0], 
                                     color=[1.0, 1.0, 1.0],
                                     scale=2.3,
                                     texture_paths=texture_paths)
            
            self.player.rotation = [0, 0, 0]
            self.using_custom_model = True
            print("✓ Modelo do jogador carregado com sucesso!")
            
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            print("→ Usando barco padrão...")
            
            boat_vertices = np.array([
                -0.4, 0, 0.6,   0.4, 0, 0.6,   0.4, 0, -0.6,  -0.4, 0, -0.6,
                -0.3, 0.5, 0.4,  0.3, 0.5, 0.4,  0.3, 0.5, -0.4, -0.3, 0.5, -0.4,
                0, 0.2, 0.8, 0, 0.3, 0.8
            ], dtype=np.float32)
            
            boat_indices = np.array([
                0,1,2, 0,2,3,  4,5,6, 4,6,7,  0,1,5, 0,5,4,
                1,2,6, 1,6,5,  2,3,7, 2,7,6,  3,0,4, 3,4,7,
                1,8,5, 0,8,4
            ], dtype=np.uint32)
            
            self.player = GameObject(boat_vertices, boat_indices, 
                                    [self.lane_positions[1], 0, 0], [0.2, 0.6, 0.9])
            self.using_custom_model = False
        
        # =========================
        # ÁGUA (PLANO GIGANTE)
        # =========================

        WATER_SIZE = 300      # tamanho no mundo
        WATER_TILE = 60       # quantas vezes a textura repete

        water_vertices = np.array([
            # posição                     cor        UV
            -WATER_SIZE, -0.5, -WATER_SIZE,  1,1,1,   0, 0,
            WATER_SIZE, -0.5, -WATER_SIZE,  1,1,1,   WATER_TILE, 0,
            WATER_SIZE, -0.5,  WATER_SIZE,  1,1,1,   WATER_TILE, WATER_TILE,
            -WATER_SIZE, -0.5,  WATER_SIZE,  1,1,1,   0, WATER_TILE,
        ], dtype=np.float32)

        water_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        self.water_vao = glGenVertexArrays(1)
        self.water_vbo = glGenBuffers(1)
        self.water_ebo = glGenBuffers(1)

        glBindVertexArray(self.water_vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.water_vbo)
        glBufferData(GL_ARRAY_BUFFER, water_vertices.nbytes, water_vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.water_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, water_indices.nbytes, water_indices, GL_STATIC_DRAW)

        # posição
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        # cor
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        # UV
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

        self.water_texture = load_texture("texture/water.jpg")

        
        # Obstáculos
        self.obstacles = []
        self.spawn_obstacle()
    
    def spawn_obstacle(self):
        lane = random.randint(0, 2)
        z_pos = -80 - random.randint(0, 30)
        
        try:
            enemy_model_path = "Boat/enemy_boat.obj"
            
            # Mesmas texturas do jogador
            enemy_texture_paths = [
                "Boat/Texture/wood1.jpg",  
                "Boat/Texture/wood2.jpg",  
                "Boat/Texture/wood3.jpg",
            ]
            
            # Cores para tingir os inimigos
            colors = [
                [1.0, 0.5, 0.5],  # Rosa
                [0.5, 1.0, 0.5],  # Verde claro
                [1.0, 1.0, 0.5],  # Amarelo claro
                [0.7, 0.5, 1.0],  # Lilás
            ]
            
            obstacle = ModelObject(
                enemy_model_path,
                [self.lane_positions[lane], 0, z_pos],
                color=random.choice(colors),
                scale=2.3,
                texture_paths=enemy_texture_paths
            )
            
            # obstacle.rotation = [0, 0, 0]
            
        except Exception as e:
            print("Erro ao carregar modelo inimigo, usando cubo:", e)
            
            obstacle_vertices = np.array([
                -0.5,-0.5,0.5,  0.5,-0.5,0.5,  0.5,0.5,0.5,  -0.5,0.5,0.5,
                -0.5,-0.5,-0.5, 0.5,-0.5,-0.5, 0.5,0.5,-0.5, -0.5,0.5,-0.5
            ], dtype=np.float32)
            
            obstacle_indices = np.array([
                0,1,2, 0,2,3,  4,5,6, 4,6,7,
                0,1,5, 0,5,4, 1,2,6, 1,6,5,
                2,3,7, 2,7,6, 3,0,4, 3,4,7
            ], dtype=np.uint32)
            
            obstacle = GameObject(
                obstacle_vertices,
                obstacle_indices,
                [self.lane_positions[lane], 0, z_pos],
                [0.9, 0.2, 0.2]
            )
        
        self.obstacles.append(obstacle)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_LEFT and self.player_lane > 0:
                self.player_lane -= 1
            elif key == glfw.KEY_RIGHT and self.player_lane < 2:
                self.player_lane += 1
            elif key == glfw.KEY_SPACE and not self.jumping:
                self.jumping = True
                self.jump_velocity = 0.15
            elif key == glfw.KEY_R and self.game_over:
                self.reset_game()
            elif key == glfw.KEY_Q:
                self.player.rotation[1] -= 90
                print(f"Rotação: {self.player.rotation}")
            elif key == glfw.KEY_E:
                self.player.rotation[1] += 90
                print(f"Rotação: {self.player.rotation}")
            elif key == glfw.KEY_Z:
                self.player.rotation = [0, 0, 0]
                print(f"Rotação resetada: {self.player.rotation}")
            elif key == glfw.KEY_EQUAL:
                self.player.scale = [s * 1.2 for s in self.player.scale]
                print(f"Escala: {self.player.scale[0]:.4f}")
            elif key == glfw.KEY_MINUS:
                self.player.scale = [s / 1.2 for s in self.player.scale]
                print(f"Escala: {self.player.scale[0]:.4f}")
    
    def reset_game(self):
        self.player_lane = 1
        self.player_y = 0.0
        self.jumping = False
        self.jump_velocity = 0.0
        self.game_speed = 0.1
        self.score = 0
        self.game_over = False
        self.obstacles.clear()
        self.spawn_obstacle()
    
    def update(self, dt):
        if self.game_over:
            return

        # mover jogador entre pistas
        target_x = self.lane_positions[self.player_lane]
        self.player.position[0] += (target_x - self.player.position[0]) * 0.20

        # pulo
        if self.jumping:
            self.player_y += self.jump_velocity
            self.jump_velocity -= 0.01
            if self.player_y <= 0:
                self.player_y = 0
                self.jumping = False
                self.jump_velocity = 0

        self.player.position[1] = self.player_y

        PLAYER_SIZE = np.array([0.8, 0.6, 1.2])
        OBSTACLE_SIZE = np.array([0.8, 0.6, 1.2])

        # obstáculos
        for obs in self.obstacles:
            obs.position[2] += self.game_speed

            if not self.jumping:
                if aabb_collision(
                    self.player.position, PLAYER_SIZE,
                    obs.position, OBSTACLE_SIZE
                ):
                    self.game_over = True
                    print(f"Game Over! Score: {self.score}")
                    return

        # remover obstáculos fora da tela
        self.obstacles = [obs for obs in self.obstacles if obs.position[2] < 10]

        if len(self.obstacles) < 3:
            self.spawn_obstacle()

        self.game_speed += 0.00005
        self.score += 1

        # print("Scala = ", obs.scale)

    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        # -------- uniforms globais --------
        time_loc = glGetUniformLocation(self.shader, "time")
        is_water_loc = glGetUniformLocation(self.shader, "isWater")
        use_texture_loc = glGetUniformLocation(self.shader, "useTexture")
        proj_loc = glGetUniformLocation(self.shader, "projection")
        view_loc = glGetUniformLocation(self.shader, "view")
        model_loc = glGetUniformLocation(self.shader, "model")

        glUniform1f(time_loc, glfw.get_time())
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, self.view)

        light_pos_loc = glGetUniformLocation(self.shader, "lightPos")
        view_pos_loc  = glGetUniformLocation(self.shader, "viewPos")
        light_color_loc = glGetUniformLocation(self.shader, "lightColor")

        glUniform3f(light_pos_loc, 5.0, 5.0, 5.0)
        glUniform3f(view_pos_loc, 0.0, 3.0, 8.0)  # mesma posição da câmera
        glUniform3f(light_color_loc, 1.0, 1.0, 1.0)


        # =====================
        # ÁGUA
        # =====================
        glUniform1i(is_water_loc, 1)
        glUniform1i(use_texture_loc, 1)

        model = pyrr.matrix44.create_identity()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        glBindTexture(GL_TEXTURE_2D, self.water.texture)
        self.water.draw()

        # =====================
        # JOGADOR
        # =====================
        glUniform1i(is_water_loc, 0)

        model = self.player.get_model_matrix()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        if hasattr(self.player, 'materials'):
            for material in self.player.materials:
                if material['texture_idx'] is not None:
                    glUniform1i(use_texture_loc, 1)
                    glBindTexture(GL_TEXTURE_2D, self.player.textures[material['texture_idx']])
                else:
                    glUniform1i(use_texture_loc, 0)

                glBindVertexArray(material['VAO'])
                glDrawElements(GL_TRIANGLES, material['indices_count'], GL_UNSIGNED_INT, None)
        else:
            glUniform1i(use_texture_loc, 0)
            self.player.draw()

        # =====================
        # OBSTÁCULOS
        # =====================
        glUniform1i(is_water_loc, 0)

        for obs in self.obstacles:
            model = obs.get_model_matrix()
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

            if hasattr(obs, 'materials'):
                for material in obs.materials:
                    if material['texture_idx'] is not None:
                        glUniform1i(use_texture_loc, 1)
                        glBindTexture(GL_TEXTURE_2D, obs.textures[material['texture_idx']])
                    else:
                        glUniform1i(use_texture_loc, 0)

                    glBindVertexArray(material['VAO'])
                    glDrawElements(GL_TRIANGLES, material['indices_count'], GL_UNSIGNED_INT, None)
            else:
                glUniform1i(use_texture_loc, 0)
                obs.draw()




                                                                    
    
    def run(self):
        last_time = glfw.get_time()
        
        print("=== NAUTIC RUNNER ===")
        print("Controles:")
        print("  SETA ESQUERDA/DIREITA - Mudar de pista")
        print("  ESPAÇO - Pular")
        print("  R - Reiniciar (após game over)")
        print("\nTeclas de DEBUG:")
        print("  Q/E - Rotacionar modelo")
        print("  Z - Resetar rotação")
        print("  +/- - Ajustar escala")
        print("\nDesvie dos obstáculos!")
        
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            glfw.poll_events()
            self.update(dt)
            self.render()
            glfw.swap_buffers(self.window)
            
            glfw.set_window_title(self.window, 
                f"Nautic Runner - BACKUP - Score: {self.score} {'[GAME OVER - Press R]' if self.game_over else ''}")
        
        glfw.terminate()

if __name__ == "__main__":
    game = NauticRunner()
    game.run()