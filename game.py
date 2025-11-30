# game_with_boat.py
# Substitui o jogador pelo modelo OBJ "boat.obj" com cor sólida.
import glfw
import OpenGL.GL as gl
import numpy as np
import math
import random
import ctypes
import sys
import os
from typing import Dict, Any, List, Tuple

# carregar o texto do arquivo do shader
def load_shader(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Shader file not found: {path}. Please ensure vertex.glsl and fragment.glsl are in the same directory.")


def check_shader_compile_status(shader_id: int, shader_type_name: str):
    """Verifica o status de compilação e lança exceção com log de erro se falhar."""
    if gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(shader_id).decode()
        # Aumenta a clareza da exceção
        gl.glDeleteShader(shader_id) 
        raise RuntimeError(f"{shader_type_name} Shader Compilation Error:\n{log}")

def check_program_link_status(program_id: int):
    """Verifica o status de linkagem e lança exceção com log de erro se falhar."""
    if gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(program_id).decode()
        gl.glDeleteProgram(program_id)
        raise RuntimeError(f"Shader Program Linkage Error:\n{log}")
    
def compile_single_shader(source: str, gl_type: int, type_name: str) -> int:
    """Cria, anexa a fonte, compila e checa o status de um único shader."""
    shader_id = gl.glCreateShader(gl_type)
    gl.glShaderSource(shader_id, source)
    gl.glCompileShader(shader_id)
    
    # Chama a função de verificação refatorada
    check_shader_compile_status(shader_id, type_name)
    
    return shader_id


def create_shader_program(vs_path: str, fs_path: str) -> int:
    """
    Carrega, compila, linka e retorna o ID de um novo programa shader.
    A função é responsável pela gestão de erros críticos.
    """
    # 1. Carregamento de Código (Fluxo de Erro Mantido)
    try:
        # Usando o nome refatorado se aplicável: vs_src = load_shader_source(vs_path)
        vs_src = load_shader(vs_path)
        fs_src = load_shader(fs_path)
    except RuntimeError as e:
        print(f"Erro de carregamento de Shader: {e}")
        # Fluxo de saída do programa, mantido do código original
        # import glfw, sys
        glfw.terminate() 
        sys.exit(1)

    # 2. Compilação (Usa a função auxiliar)
    vs_id = compile_single_shader(vs_src, gl.GL_VERTEX_SHADER, "Vertex")
    fs_id = compile_single_shader(fs_src, gl.GL_FRAGMENT_SHADER, "Fragment")

    # 3. Linkagem
    program_id = gl.glCreateProgram()
    gl.glAttachShader(program_id, vs_id)
    gl.glAttachShader(program_id, fs_id)
    gl.glLinkProgram(program_id)

    # 4. Verificação de Linkagem (Usa a função auxiliar)
    check_program_link_status(program_id)

    # 5. Limpeza de Recursos
    gl.glDeleteShader(vs_id)
    gl.glDeleteShader(fs_id)

    return program_id


def _parse_obj_data(path: str) -> tuple[list[list[float]], list[list[int]]]:
    """
    Responsabilidade: Apenas lê o arquivo OBJ e extrai vértices e faces (índices v).
    Retorna: (lista de vértices brutos, lista de faces trianguladas)
    """
    raw_verts = []
    raw_faces = []

    print(f"Carregando {path}...")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                try:
                    # Lê 3 coordenadas
                    _, x, y, z = line.split()[:4]
                    raw_verts.append([float(x), float(y), float(z)])
                except ValueError:
                    continue

            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = []
                
                # Extração e Correção de Índices
                for p in parts:
                    try:
                        # Extrai o primeiro índice (índice do vértice 'v')
                        v_idx = p.split("/")[0]
                        # OBJ usa 1-based indexing, então subtraímos 1
                        idxs.append(int(v_idx) - 1)
                    except (ValueError, IndexError):
                        continue

                # Triangulação (Fan Triangulation) para faces com 4 ou mais vértices
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        raw_faces.append([idxs[0], idxs[i], idxs[i+1]])

    if len(raw_verts) == 0:
        raise RuntimeError(f"OBJ sem vértices úteis (v ).")
    if len(raw_faces) == 0:
        raise RuntimeError(f"OBJ sem faces úteis (f ).")

    return raw_verts, raw_faces

def load_obj_simple(path: str, target_size: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega um arquivo OBJ, centraliza e escala a geometria.
    Retorna um array de vértices e um array 1D de índices.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")

    # 1. Parsing de Dados (Responsabilidade da função auxiliar)
    verts_list, faces_list = _parse_obj_data(path)

    # 2. Conversão para NumPy
    vertices = np.array(verts_list, dtype=np.float32)
    indices = np.array(faces_list, dtype=np.uint32).ravel()
    
    # 3. Pré-processamento: Centralização e Escalonamento (Função bem definida)
    
    # Cálculo Bounding Box
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    
    # Centralização
    center = (vmin + vmax) / 2.0
    vertices -= center # Subtração element-wise

    # Escalonamento
    extents = vmax - vmin
    max_extent = np.max(extents) # Usando np.max para clareza
    
    if max_extent > 0:
        scale = target_size / max_extent
        vertices *= scale
    else:
        scale = 1.0

    # 4. Debug e Retorno
    
    num_tris = len(indices) // 3
    print(f"[load_obj_simple] {path}: verts={len(vertices)}, tris={num_tris}")
    print(f"[DEBUG] bbox min/max (após escala) = {vertices.min(axis=0)}, {vertices.max(axis=0)}")
    print(f"[DEBUG] scale usado: {scale}")

    return vertices, indices


def create_perspective_matrix(fov_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    """
    Calcula e retorna a Matriz de Projeção em Perspectiva (OpenGL/Right-Handed System).

    Parâmetros:
      fov_rad (float): Campo de visão vertical em RADIANOS.
      aspect (float): Relação de aspecto (largura / altura).
      near (float): Distância do plano de corte próximo (near plane).
      far (float): Distância do plano de corte distante (far plane).
    """
    # 1. Tratamento de Erros de Entrada
    if near == far or near <= 0 or far <= 0 or fov_rad <= 0:
        raise ValueError("Parâmetros de projeção inválidos. 'near', 'far', e 'fov' devem ser positivos, e 'near' != 'far'.")

    # 2. Cálculo do Foco (f)
    # f = cot(fov/2), que é 1 / tan(fov/2)
    tan_half_fov = math.tan(fov_rad / 2.0)
    if tan_half_fov == 0:
        # Prevê divisão por zero se fov for 0 ou 180 (o que é impossível)
        raise ValueError("FOV inválido, tan(fov/2) é zero.")
        
    f = 1.0 / tan_half_fov
    
    # 3. Denominador Comum (para os elementos da 3ª linha)
    range_inv = 1.0 / (near - far)

    # 4. Construção da Matriz (Usando variáveis nomeadas para clareza)
    
    # Elementos A e B (foco e razão de aspecto)
    A = f / aspect
    B = f
    
    # Elementos C e D (profundidade - Z)
    # C = (far + near) / (near - far) 
    # D = (2 * far * near) / (near - far)
    C = (far + near) * range_inv
    D = (2.0 * far * near) * range_inv 

    mat = np.array([
        [A, 0.0, 0.0, 0.0],
        [0.0, B, 0.0, 0.0],
        [0.0, 0.0, C, D],
        [0.0, 0.0, -1.0, 0.0], # Linha 4: W = -Z (divisão em perspectiva)
    ], dtype=np.float32)
    
    return mat


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normaliza um vetor NumPy. Lança erro se o vetor for nulo."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Vetor nulo não pode ser normalizado na função look_at.")
    return v / norm

def create_view_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Calcula e retorna a Matriz View (Câmera).
    
    Parâmetros:
      eye (np.ndarray): Posição da Câmera (ponto de observação).
      center (np.ndarray): Ponto para onde a câmera está olhando (target).
      up (np.ndarray): Vetor 'para cima' global (geralmente [0, 1, 0]).
    """
    # 1. Calcular o Vetor de Orientação (Forward, F)
    # Vetor F (direção Z da câmera)
    f = center - eye
    # Normalização única e segura do vetor F
    f = _normalize_vector(f) 
    
    # 2. Calcular o Vetor Lateral (Side, S)
    # O 'up' original deve ser normalizado antes do cross product
    u_norm = _normalize_vector(up) 
    
    # Vetor S (direção X da câmera)
    s = np.cross(f, u_norm)
    # O resultado do cross product deve ser normalizado
    s = _normalize_vector(s) 
    
    # 3. Calcular o Vetor "Para Cima" Corrigido (Up, U)
    # Vetor U (direção Y da câmera)
    # Recalculamos U como o cross product de S e F.
    # U já será ortogonal a F e S, e já estará normalizado se F e S estiverem.
    u = np.cross(s, f) 
    
    # 4. Construção da Matriz de Orientação (Rotação) e Translação
    
    # Matriz M (Matriz de Rotação 3x3 + Translação)
    # (A translação é aplicada por último, usando o produto escalar)
    M = np.identity(4, dtype=np.float32)

    # Colocando os vetores de base S, U, -F nas primeiras 3 linhas/colunas
    # Nota: M[2, :3] = -f é o vetor 'Backward'
    M[0, :3] = s  # X-Axis (Direita)
    M[1, :3] = u  # Y-Axis (Cima)
    M[2, :3] = -f # Z-Axis (Para trás)

    # Translação: Aplica a rotação inversa à posição da câmera ('eye')
    # O termo de translação é: - (M_rotação * eye)
    M[:3, 3] = -np.dot(M[:3, :3], eye)

    return M

# --------------------------
# CUBO (para obstáculos)
# --------------------------
cube_vertices = np.array([
    # Z positiva (Frente)
    -0.5, -0.5,  0.5,  # 0: Canto Inferior Esquerdo Frontal
     0.5, -0.5,  0.5,  # 1: Canto Inferior Direito Frontal
     0.5,  0.5,  0.5,  # 2: Canto Superior Direito Frontal
    -0.5,  0.5,  0.5,  # 3: Canto Superior Esquerdo Frontal
    
    # Z negativa (Trás)
    -0.5, -0.5, -0.5,  # 4: Canto Inferior Esquerdo Traseiro
     0.5, -0.5, -0.5,  # 5: Canto Inferior Direito Traseiro
     0.5,  0.5, -0.5,  # 6: Canto Superior Direito Traseiro
    -0.5,  0.5, -0.5,  # 7: Canto Superior Esquerdo Traseiro
], dtype=np.float32)

cube_indices = np.array([
    # Face 0: Frente (Z+) (Vértices 0, 1, 2, 3)
    0, 1, 2,    # Triângulo 1 (0-1-2)
    2, 3, 0,    # Triângulo 2 (2-3-0)
    
    # Face 1: Direita (X+) (Vértices 1, 5, 6, 2)
    1, 5, 6,    # Triângulo 3 (1-5-6)
    6, 2, 1,    # Triângulo 4 (6-2-1)
    
    # Face 2: Trás (Z-) (Vértices 5, 4, 7, 6)
    5, 4, 7,    # Triângulo 5 (5-4-7)
    7, 6, 5,    # Triângulo 6 (7-6-5)
    
    # Face 3: Esquerda (X-) (Vértices 4, 0, 3, 7)
    4, 0, 3,    # Triângulo 7 (4-0-3)
    3, 7, 4,    # Triângulo 8 (3-7-4)
    
    # Face 4: Topo (Y+) (Vértices 3, 2, 6, 7)
    3, 2, 6,    # Triângulo 9 (3-2-6)
    6, 7, 3,    # Triângulo 10 (6-7-3)
    
    # Face 5: Base (Y-) (Vértices 4, 5, 1, 0)
    4, 5, 1,    # Triângulo 11 (4-5-1)
    1, 0, 4     # Triângulo 12 (1-0-4)
], dtype=np.uint32)


TRACK_Y_LEVEL = -3.0 # Constante: Nível Y onde o chão/água está posicionado

# Definição dos Vértices do Plano (Quadrado)
# Ordem: (X, Y, Z)
track_vertices = np.array([
    # X     Y           Z
    -3.0, TRACK_Y_LEVEL,   5.0,  # 0: Canto Frontal Esquerdo (Perto da Câmera)
     3.0, TRACK_Y_LEVEL,   5.0,  # 1: Canto Frontal Direito
     3.0, TRACK_Y_LEVEL, -200.0, # 2: Canto Distante Direito
    -3.0, TRACK_Y_LEVEL, -200.0, # 3: Canto Distante Esquerdo
], dtype=np.float32)

# Definição dos Índices (Dois Triângulos formam o Quadrado)
track_indices = np.array([
    0, 1, 2,  # Primeiro Triângulo (0 -> 1 -> 2)
    2, 3, 0   # Segundo Triângulo (2 -> 3 -> 0)
], dtype=np.uint32)

import numpy as np
# Assumindo que TRACK_Y_LEVEL foi substituído pelo parâmetro 'y_level'

def create_ground_plane(width: float, depth: float, y_level: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria a geometria (vértices e índices) de um plano centrado no eixo X,
    com dimensões 'width' (X) e 'depth' (Z) na altura 'y_level' (Y).
    
    Parâmetros:
      width (float): Largura total (de -width/2 a +width/2 no X).
      depth (float): Profundidade total (de -depth/2 a +depth/2 no Z).
      y_level (float): Altura Y do plano.
    """
    half_w = width / 2.0
    half_d = depth / 2.0

    vertices = np.array([
        # X           Y           Z
        -half_w, y_level,  half_d,  # 0: Frontal Esquerdo
         half_w, y_level,  half_d,  # 1: Frontal Direito
         half_w, y_level, -half_d,  # 2: Distante Direito
        -half_w, y_level, -half_d,  # 3: Distante Esquerdo
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,  # T1
        2, 3, 0   # T2
    ], dtype=np.uint32)
    
    return vertices, indices

# Exemplo de uso (para replicar o código original):
# O código original tem -3.0 a 3.0 em X (width=6) e 5.0 a -200.0 em Z (depth=205)
# No entanto, a função acima cria um plano *centrado*. 
# Se você deseja manter a posição exata:

def create_custom_plane(x_min, x_max, z_min, z_max, y_level) -> tuple[np.ndarray, np.ndarray]:
    """Cria um plano baseado em coordenadas de canto (não centrado)."""
    vertices = np.array([
        [x_min, y_level, z_max],  # 0: Canto Frontal Esquerdo (max Z)
        [x_max, y_level, z_max],  # 1: Canto Frontal Direito
        [x_max, y_level, z_min],  # 2: Canto Distante Direito (min Z)
        [x_min, y_level, z_min],  # 3: Canto Distante Esquerdo
    ], dtype=np.float32).flatten() # Flatten para o formato 1D

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    return vertices, indices

# Uso do código original com a nova função:
# track_vertices, track_indices = create_custom_plane(-3.0, 3.0, -200.0, 5.0, TRACK_Y_LEVEL)

# Constante, pode permanecer global se for fixa
LANE_POSITIONS = [-2.0, 0.0, 2.0] # Renomeado e tipado para float

def initialize_game_state() -> Dict[str, Any]:
    """
    Inicializa e retorna o dicionário contendo o estado principal do jogo.
    """
    # Usamos o índice do array LANE_POSITIONS (0, 1, 2)
    # 1 corresponde à posição central (0.0)
    INITIAL_LANE_INDEX = 1 
    
    # O estado é um dicionário que será passado para as funções de update e draw
    game_state = {
        # Configuração de Pista
        "lane_positions": LANE_POSITIONS,   # Posições X das pistas
        "current_lane_index": INITIAL_LANE_INDEX, # Índice atual da pista (0, 1, ou 2)
        
        # Objetos Dinâmicos
        "obstacles": [],                   # Lista para armazenar objetos/obstáculos
        
        # Gerenciamento de Tempo/Eventos
        "spawn_timer": 0.0,                # Contador de tempo para spawn de novos obstáculos
        
        # Estado de Entrada (Input)
        "last_left_input": False,          # Se o input "esquerda" foi pressionado/tratado no frame anterior
        "last_right_input": False,         # Se o input "direita" foi pressionado/tratado no frame anterior
        
        # Adicione outros estados necessários, como "game_over": False, "score": 0, etc.
    }
    return game_state


# --------------------------
# DESENHO
# --------------------------
# --------------------------
# DESENHO (AGORA COM ROTAÇÃO)
# --------------------------
def create_model_matrix(x: float, y: float, z: float, scale: float, rotation_y: float = 0.0) -> np.ndarray:
    """
    Constrói a Matriz Model 4x4 aplicando: Rotação Y -> Escala -> Translação.
    
    Parâmetros:
      x, y, z (float): Posição da translação.
      scale (float): Fator de escala uniforme.
      rotation_y (float): Ângulo de rotação em torno do eixo Y (em radianos).
    """
    # 1. Matriz de Transformação Mestra (começa como Identidade)
    model = np.identity(4, dtype=np.float32)

    # 2. Escala (Aplicada ANTES da rotação e translação)
    # Note: A forma mais limpa de aplicar escala é multiplicando os elementos da diagonal:
    model[0, 0] *= scale
    model[1, 1] *= scale
    model[2, 2] *= scale

    # 3. Rotação Y (Cálculo e Multiplicação)
    c = math.cos(rotation_y)
    s = math.sin(rotation_y)
    
    R_y = np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)

    # Aplica Rotação à matriz (R * M), se o Model já tiver sido inicializado com a Escala
    # O ordem é crucial: Rotação * Escala * Identidade
    model = np.dot(R_y, model) 

    # 4. Translação (Aplicada por último, nos últimos elementos da matriz)
    # Translação é sempre o último passo na matriz Model
    model[0, 3] = x
    model[1, 3] = y
    model[2, 3] = z
    
    return model

def render_mesh(shader_id: int, vao_id: int, index_count: int, 
                model_matrix: np.ndarray, color: Tuple[float, float, float], 
                view_matrix: np.ndarray, proj_matrix: np.ndarray):
    """
    Define uniforms, anexa a geometria e executa a chamada de desenho.
    A Matriz Model deve ser pré-calculada e fornecida.
    """
    gl.glUseProgram(shader_id)

    # O método .T (Transposta) é usado no Python/NumPy para adequar o formato à 
    # convenção de OpenGL, onde as matrizes são lidas em Column-Major Order.

    # 1. Definir Uniforms de Matriz (MVPs)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "model"), 
                          1, gl.GL_FALSE, model_matrix.T)
                          
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "view"), 
                          1, gl.GL_FALSE, view_matrix.T)
                          
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "projection"), 
                          1, gl.GL_FALSE, proj_matrix.T)

    # 2. Definir Uniform de Cor (assumindo objColor no shader)
    gl.glUniform3f(gl.glGetUniformLocation(shader_id, "objColor"), *color)
    
    # 3. Chamada de Desenho (Draw Call)
    gl.glBindVertexArray(vao_id)
    gl.glDrawElements(gl.GL_TRIANGLES, index_count, gl.GL_UNSIGNED_INT, None)
    gl.glBindVertexArray(0) # Desvincula o VAO


# --------------------------
# MAIN
# --------------------------
# Função refatorada para a criação de buffers de qualquer mesh
def setup_mesh_buffers(vertices: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
    """
    Cria e configura o VAO, VBO e EBO para um mesh dado.
    Assume que o Attrib Pointer 0 (posição) é o único usado.
    Retorna: (VAO ID, Número de Índices)
    """

    if indices.size == 0 or vertices.size == 0:
        print(f"[ERROR] Mesh Buffer Setup Falhou: Vertices ou Índices vazios.")
        return 0, 0 # Retorna 0 para VAO e 0 para a contagem, forçando o else!
    
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # VBO
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    # EBO (Element Buffer Object)
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    # Configuração do Atributo 0 (Posição XYZ)
    gl.glEnableVertexAttribArray(0)
    # 3 componentes (XYZ), Tipo Float, 12 bytes por vértice (3*4)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, None)

    gl.glBindVertexArray(0)
    return vao, len(indices)

def handle_input(win: glfw._GLFWwindow, game_state: Dict[str, Any]):
    """Atualiza o índice da pista baseado no input do usuário e gerencia o anti-spam."""
    
    current_lane = game_state["current_lane_index"]
    last_left = game_state["last_left_input"]
    last_right = game_state["last_right_input"]
    
    # Mapeamento de teclas (A e Esquerda, D e Direita)
    left = glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_A) == glfw.PRESS
    right = glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS or \
            glfw.get_key(win, glfw.KEY_D) == glfw.PRESS

    # Lógica de Troca de Pista (anti-spam usando borda ascendente)
    if left and not last_left:  
        current_lane = max(0, current_lane - 1)
    if right and not last_right:
        current_lane = min(2, current_lane + 1)
    
    # Atualiza o estado para o próximo frame
    game_state["current_lane_index"] = current_lane
    game_state["last_left_input"] = left
    game_state["last_right_input"] = right

# A função update_game lida com a lógica de objetos e colisão
def update_game(win: glfw._GLFWwindow, game_state: Dict[str, Any], delta_time: float = 1.0):
    """
    Atualiza o estado de obstáculos, spawns e verifica colisões.
    (delta_time adicionado para potencial future frame-rate independence)
    """
    
    # 1. Spawn de Obstáculos
    game_state["spawn_timer"] += delta_time
    if game_state["spawn_timer"] > 30: # 30 "ticks" (dependente do framerate atual)
        game_state["spawn_timer"] = 0.0
        # Obstáculo: [índice da pista, nível Y, posição Z]
        game_state["obstacles"].append([random.choice([0,1,2]), -3.0, -40.0]) # Usando -3.0 como altura base
    
    # 2. Movimentação e Colisão
    new_obs = []
    player_lane = game_state["current_lane_index"]
    
    for lane, oy, oz in game_state["obstacles"]:
        oz += 0.3 * delta_time # Velocidade
        
        # Colisão
        if abs(oz) < 0.9 and lane == player_lane:
            print("COLISÃO! Fim de jogo.")
            glfw.set_window_should_close(win, True)
            
        # Manter obstáculo se estiver na tela
        if oz < 20: 
            new_obs.append([lane, oy, oz])

    game_state["obstacles"] = new_obs

def main():
    # Inicializa o estado do jogo (substitui as globais soltas)
    # Assumindo que initialize_game_state() foi refatorada
    game_state = initialize_game_state() 
    lanes = game_state["lane_positions"] # Atalho para as posições X
    
    if not glfw.init():
        raise RuntimeError("glfw.init failed")

    win = None
    try:
        # -----------------------
        # 1. INICIALIZAÇÃO GLFW & OPENGL
        # -----------------------
        win = glfw.create_window(900, 600, "Boat as Player", None, None)
        if not win:
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(win)
        gl.glEnable(gl.GL_DEPTH_TEST)
        shader = create_shader_program("vertex.glsl","fragment.glsl")

        # -----------------------
        # 2. CONFIGURAÇÃO DE MESHS
        # -----------------------
        
        # Pista
        track_vao, track_index_count = setup_mesh_buffers(track_vertices, track_indices)

        # Cubo (Obstáculos e Substituto do Jogador)
        cube_vao, cube_index_count = setup_mesh_buffers(cube_vertices, cube_indices)

        # Jogador (OBJ)
        model_vao, model_index_count = 0, 0 # Inicializar com 0
        try:
            model_vertices, model_indices = load_obj_simple("Medieval Boat.obj", target_size=100.0)
            model_vao, model_index_count = setup_mesh_buffers(model_vertices, model_indices)
            
            # --- NOVO DEBUG: FORÇAR CONTAGEM SE O VAO FOR CRIADO ---
            if model_vao != 0 and model_index_count == 0:
                # Se o VAO foi criado (ID > 0), mas a contagem é 0, force a contagem para 1.
                # Isso garante que a lógica 'if model_index_count > 0' seja TRUE.
                model_index_count = 1 
            # -----------------------------------------------------

            print("[main] Modelo OBJ carregado com sucesso.")
        except Exception as e:
            print(f"ERRO ao carregar OBJ: {e}. Usando cubo substituto.")

        # -----------------------
        # 3. GAME LOOP
        # -----------------------
        while not glfw.window_should_close(win):
            # 3.1. Pré-Desenho
            gl.glClearColor(0,0,0,1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            # Câmera
            view = create_view_matrix(
                np.array([0,2,7],np.float32), np.array([0,0,0],np.float32), np.array([0,1,0],np.float32)
            )
            width, height = glfw.get_framebuffer_size(win)
            proj = create_perspective_matrix(math.radians(60), width/height, 0.1, 200)

            # 3.2. Lógica do Jogo
            handle_input(win, game_state)
            update_game(win, game_state)
            
            # Posições atualizadas
            player_x = lanes[game_state["current_lane_index"]]
            player_y = TRACK_Y_LEVEL + 0.5
            
            # 3.3. DESENHO
            
            # Desenha Pista
            track_model_mat = create_model_matrix(0, 0, 0, 1) # Argumento 4: Matriz Model
            render_mesh(
                shader_id=shader, 
                vao_id=track_vao, 
                index_count=track_index_count, 
                model_matrix=track_model_mat,           # <--- Matriz Model
                color=(0.1, 0.2, 1), 
                view_matrix=view, 
                proj_matrix=proj
            )

            # Desenha Jogador (Barco ou Cubo Substituto)
            if model_index_count > 0:
                vao_to_draw, count_to_draw, color = model_vao, model_index_count, (1,0.2,0.2)
            else:
                vao_to_draw, count_to_draw, color = cube_vao, cube_index_count, (1, 0.0, 0.0)

            player_model_mat = create_model_matrix(
                x=player_x, 
                y=player_y, 
                z=0.0, 
                scale=1.0, 
                rotation_y=math.radians(180.0)
            )
            render_mesh(
                shader_id=shader, 
                vao_id=vao_to_draw, 
                index_count=count_to_draw, 
                model_matrix=player_model_mat,          # <--- Matriz Model
                color=color, 
                view_matrix=view, 
                proj_matrix=proj
            ) 

            # Desenha Obstáculos
            for lane_idx, oy, oz in game_state["obstacles"]:
                obs_model_mat = create_model_matrix(
                    x=lanes[lane_idx], 
                    y=oy + 1.0, 
                    z=oz, 
                    scale=1.0,
                    rotation_y=0.0 # Sem rotação
                )
                render_mesh(
                    shader_id=shader, 
                    vao_id=cube_vao, 
                    index_count=cube_index_count, 
                    model_matrix=obs_model_mat,          # <--- Matriz Model
                    color=(0, 0.9, 0.2), 
                    view_matrix=view, 
                    proj_matrix=proj
                )

            glfw.swap_buffers(win)
            glfw.poll_events()

    except Exception as e:
        print(f"Ocorreu um erro fatal: {e}")
        
    finally:
        if win:
            glfw.destroy_window(win)
        glfw.terminate()

if __name__ == "__main__":
    main()