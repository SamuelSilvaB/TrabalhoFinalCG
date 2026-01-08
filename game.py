import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from math import sin, cos, radians
import random

# Vertex Shader
vertex_src = """
# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# Fragment Shader
fragment_src = """
# version 330 core

in vec3 v_color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(v_color, 1.0);
}
"""

class GameObject:
    def __init__(self, vertices, indices, position, color):
        self.position = np.array(position, dtype=np.float32)
        self.vertices = vertices
        self.indices = indices
        self.color = color
        self.rotation = [0, 0, 0]  # rotação em x, y, z
        self.scale = [1, 1, 1]  # escala em x, y, z
        self.setup_mesh()
    
    def setup_mesh(self):
        colored_vertices = []
        for i in range(0, len(self.vertices), 3):
            colored_vertices.extend([self.vertices[i], self.vertices[i+1], self.vertices[i+2]])
            colored_vertices.extend(self.color)
        
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)
        
        glBindVertexArray(self.VAO)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(colored_vertices, dtype=np.float32).nbytes,
                     np.array(colored_vertices, dtype=np.float32), GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    
    def get_model_matrix(self):
        # Criar matriz de transformação
        model = pyrr.matrix44.create_identity()
        
        # Translação
        model = pyrr.matrix44.multiply(model, 
            pyrr.matrix44.create_from_translation(self.position))
        
        # Rotação (Y -> X -> Z)
        if self.rotation[1] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_y_rotation(radians(self.rotation[1])))
        if self.rotation[0] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_x_rotation(radians(self.rotation[0])))
        if self.rotation[2] != 0:
            model = pyrr.matrix44.multiply(model,
                pyrr.matrix44.create_from_z_rotation(radians(self.rotation[2])))
        
        # Escala
        model = pyrr.matrix44.multiply(model,
            pyrr.matrix44.create_from_scale(self.scale))
        
        return model
    
    def draw(self):
        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

class ModelObject:
    """Classe para carregar modelos .obj manualmente"""
    def __init__(self, obj_path, position, color=[0.7, 0.7, 0.7], scale=1.0):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = [0, 0, 0]
        self.scale = [scale, scale, scale]
        self.color = color
        
        # Carregar modelo .obj manualmente
        vertices, indices = self.load_obj(obj_path)
        self.vertices = vertices
        self.indices = indices
        self.setup_mesh()
    
    def load_obj(self, filename):
        """Carrega arquivo .obj manualmente"""
        vertices = []
        faces = []
        
        print(f"Carregando modelo: {filename}")
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    
                    # Vértices (v x y z)
                    if parts[0] == 'v':
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])
                    
                    # Faces (f v1 v2 v3 ou f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                    elif parts[0] == 'f':
                        face = []
                        for i in range(1, len(parts)):
                            # Pegar apenas o índice do vértice (ignora textura e normal)
                            vertex_index = parts[i].split('/')[0]
                            face.append(int(vertex_index) - 1)  # OBJ usa índice 1-based
                        
                        # Triangular faces com mais de 3 vértices
                        for i in range(1, len(face) - 1):
                            faces.append([face[0], face[i], face[i + 1]])
            
            print(f"✓ Carregados {len(vertices)} vértices e {len(faces)} faces")
            
            # Calcular centro e tamanho do modelo para normalização
            if vertices:
                vertices_np = np.array(vertices)
                min_coords = vertices_np.min(axis=0)
                max_coords = vertices_np.max(axis=0)
                center = (min_coords + max_coords) / 2
                size = max_coords - min_coords
                max_size = max(size)
                
                print(f"  Centro: {center}")
                print(f"  Tamanho: {size}")
                print(f"  Maior dimensão: {max_size}")
                
                # Centralizar modelo
                # Centralizar e NORMALIZAR modelo
                
                for i in range(len(vertices)):
                    vertices[i][0] = (vertices[i][0] - center[0]) / max_size
                    vertices[i][1] = (vertices[i][1] - center[1]) / max_size
                    vertices[i][2] = (vertices[i][2] - center[2]) / max_size

            
            # Converter para arrays numpy
            vertices_array = []
            indices_array = []
            
            for face in faces:
                for vertex_idx in face:
                    if vertex_idx < len(vertices):
                        vertices_array.extend(vertices[vertex_idx])
                        vertices_array.extend(self.color)
                        indices_array.append(len(indices_array))
            
            return (np.array(vertices_array, dtype=np.float32),
                    np.array(indices_array, dtype=np.uint32))
        
        except FileNotFoundError:
            print(f"✗ Arquivo não encontrado: {filename}")
            raise
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            raise
    
    def setup_mesh(self):
        self.VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)
        
        glBindVertexArray(self.VAO)
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        # Posição (x, y, z)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        
        # Cor (r, g, b)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    
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
        
        self.shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        
        # Game state
        self.player_lane = 1  # 0=esquerda, 1=meio, 2=direita
        self.player_y = 0.0
        self.jumping = False
        self.jump_velocity = 0.0
        self.game_speed = 0.1
        self.score = 0
        self.game_over = False
        self.using_custom_model = False  # Flag para saber se está usando modelo customizado
        
        # Configuração das pistas (lanes)
        self.lane_positions = [-2.0, 0.0, 2.0]
        
        self.setup_game_objects()
        
        # Projection matrix
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.width/self.height, 0.1, 100
        )
        
        # View matrix (câmera atrás do jogador)
        self.view = pyrr.matrix44.create_look_at(
            pyrr.Vector3([0, 3, 8]),
            pyrr.Vector3([0, 0, 0]),
            pyrr.Vector3([0, 1, 0])
        )
    
    def setup_game_objects(self):
        # Tentar carregar modelo externo, caso contrário usar barco padrão
        try:
            # ALTERE ESTE CAMINHO PARA O SEU ARQUIVO .OBJ
            model_path = "Boat/boat.obj"  # <-- Coloque o caminho do seu modelo aqui
            
            self.player = ModelObject(model_path, 
                                     [self.lane_positions[1], 0, 0], 
                                     color=[0.6, 0.4, 0.2],  # Cor marrom para barco medieval
                                     scale=1.3)  # Escala bem pequena - ajuste conforme necessário
            
            # Ajuste a rotação se necessário
            # Se os controles estiverem invertidos, ajuste aqui:
            # - Controles normais: rotation = [0, 0, 0]
            # - Barco virado 180°: rotation = [0, 180, 0]
            # - Barco de lado: rotation = [0, 90, 0] ou [0, 270, 0]
            self.player.rotation = [0, 0, 0]  # <-- AJUSTE AQUI se necessário
            
            self.using_custom_model = True
            print("✓ Modelo externo carregado com sucesso!")
            print(f"→ Se os controles estiverem invertidos, ajuste 'rotation' em setup_game_objects()")
            
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            print("→ Usando barco padrão...")
            
            # Barco padrão (caso o modelo não carregue)
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
        
        # Água (plano)
        water_vertices = np.array([
            -10, -0.5, -100,  10, -0.5, -100,
            10, -0.5, 100,   -10, -0.5, 100
        ], dtype=np.float32)
        
        water_indices = np.array([0,1,2, 0,2,3], dtype=np.uint32)
        self.water = GameObject(water_vertices, water_indices, [0,0,0], [0.0, 0.4, 0.7])
        
        # Obstáculos (rochas/boias)
        self.obstacles = []
        self.spawn_obstacle()
    
    def spawn_obstacle(self):
        lane = random.randint(0, 2)
        z_pos = 80 + random.randint(0, 30)

        try:
            enemy_model_path = "Boat/boat.obj"  # <-- ajuste o caminho

            obstacle = ModelObject(
                enemy_model_path,
                [self.lane_positions[lane], 0, z_pos],
                color=[0.7, 0.2, 0.2],   # vermelho inimigo
                scale=1.2               # ajuste fino
            )

            # Ajuste de orientação se necessário
            obstacle.rotation = [0, 180, 0]

        except Exception as e:
            print("Erro ao carregar modelo inimigo, usando cubo:", e)

            # fallback (se der erro)
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
            
            # Teclas de DEBUG para ajustar rotação do modelo em tempo real
            elif key == glfw.KEY_Q:  # Q - Rotacionar -90° em Y
                self.player.rotation[1] -= 90
                print(f"Rotação: {self.player.rotation}")
            elif key == glfw.KEY_E:  # E - Rotacionar +90° em Y
                self.player.rotation[1] += 90
                print(f"Rotação: {self.player.rotation}")
            elif key == glfw.KEY_Z:  # Z - Resetar rotação
                self.player.rotation = [0, 0, 0]
                print(f"Rotação resetada: {self.player.rotation}")
            
            # Teclas de DEBUG para ajustar escala
            elif key == glfw.KEY_EQUAL:  # + Aumentar escala
                self.player.scale = [s * 1.2 for s in self.player.scale]
                print(f"Escala: {self.player.scale[0]:.4f}")
            elif key == glfw.KEY_MINUS:  # - Diminuir escala
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
        
        # Atualizar posição do jogador
        target_x = self.lane_positions[self.player_lane]
        self.player.position[0] += (target_x - self.player.position[0]) * 0.20
        
        # Pulo
        if self.jumping:
            self.player_y += self.jump_velocity
            self.jump_velocity -= 0.01
            if self.player_y <= 0:
                self.player_y = 0
                self.jumping = False
                self.jump_velocity = 0
        
        self.player.position[1] = self.player_y
        
        # Mover obstáculos
        for obs in self.obstacles:
            obs.position[2] -= self.game_speed

            distance = obs.position[2]
            scale = max(0.5, min(1.5, 1.5 - distance * 0.01))
            obs.scale = [scale, scale, scale]

            # Verificar colisão
            if not self.jumping and abs(obs.position[2]) < 1.5:
                if abs(self.player.position[0] - obs.position[0]) < 0.8:
                    self.game_over = True
                    print(f"Game Over! Score: {self.score}")
        
        # Remover obstáculos que passaram e adicionar novos
        self.obstacles = [obs for obs in self.obstacles if obs.position[2] > -10]
        
        if len(self.obstacles) < 3:
            self.spawn_obstacle()
        
        # Aumentar dificuldade
        self.game_speed += 0.00005
        self.score += 1
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        
        proj_loc = glGetUniformLocation(self.shader, "projection")
        view_loc = glGetUniformLocation(self.shader, "view")
        model_loc = glGetUniformLocation(self.shader, "model")
        
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, self.view)
        
        # Desenhar água
        model = self.water.get_model_matrix()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        self.water.draw()
        
        # Desenhar jogador
        model = self.player.get_model_matrix()
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        self.player.draw()
        
        # Desenhar obstáculos
        for obs in self.obstacles:
            model = obs.get_model_matrix()
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            obs.draw()
    
    def run(self):
        last_time = glfw.get_time()
        
        print("=== NAUTIC RUNNER ===")
        print("Controles:")
        print("  SETA ESQUERDA/DIREITA - Mudar de pista")
        print("  ESPAÇO - Pular")
        print("  R - Reiniciar (após game over)")
        print("\nTeclas de DEBUG:")
        print("  Q/E - Rotacionar modelo (testar orientação)")
        print("  Z - Resetar rotação")
        print("  +/- - Ajustar escala do modelo")
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
                f"Nautic Runner - Score: {self.score} {'[GAME OVER - Press R]' if self.game_over else ''}")
        
        glfw.terminate()

if __name__ == "__main__":
    game = NauticRunner()
    game.run()