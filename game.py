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
        
        gl.glDeleteShader(shader_id) 
        raise RuntimeError(f"{shader_type_name} Shader Compilation Error:\n{log}")

def check_program_link_status(program_id: int):
    if gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(program_id).decode()
        gl.glDeleteProgram(program_id)
        raise RuntimeError(f"Shader Program Linkage Error:\n{log}")
    
def compile_single_shader(source: str, gl_type: int, type_name: str) -> int:
    shader_id = gl.glCreateShader(gl_type)
    gl.glShaderSource(shader_id, source)
    gl.glCompileShader(shader_id)
    
    check_shader_compile_status(shader_id, type_name)
    
    return shader_id


def create_shader_program(vs_path: str, fs_path: str) -> int:
    """
    Carrega, compila, linka e retorna o ID de um novo programa shader.
    A função é responsável pela gestão de erros críticos.
    """
    # Carregamento de Código (Fluxo de Erro Mantido)
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

    # Compilação (Usa a função auxiliar)
    vs_id = compile_single_shader(vs_src, gl.GL_VERTEX_SHADER, "Vertex")
    fs_id = compile_single_shader(fs_src, gl.GL_FRAGMENT_SHADER, "Fragment")

    # Linkagem
    program_id = gl.glCreateProgram()
    gl.glAttachShader(program_id, vs_id)
    gl.glAttachShader(program_id, fs_id)
    gl.glLinkProgram(program_id)

    # Verificação de Linkagem (Usa a função auxiliar)
    check_program_link_status(program_id)

    # Limpeza de Recursos
    gl.glDeleteShader(vs_id)
    gl.glDeleteShader(fs_id)

    return program_id


def _parse_obj_data(path: str) -> tuple[list[list[float]], list[list[int]]]:
    
    raw_verts = []
    raw_faces = []

    print(f"Carregando {path}...")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                try:
                    _, x, y, z = line.split()[:4]
                    raw_verts.append([float(x), float(y), float(z)])
                except ValueError:
                    continue

            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = []
                
                for p in parts:
                    try:
                        # Extrai o primeiro índice (índice do vértice 'v')
                        v_idx = p.split("/")[0]
                        # OBJ usa 1-based indexing, então subtraímos 1
                        idxs.append(int(v_idx) - 1)
                    except (ValueError, IndexError):
                        continue

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

    verts_list, faces_list = _parse_obj_data(path)

    vertices = np.array(verts_list, dtype=np.float32)
    indices = np.array(faces_list, dtype=np.uint32).ravel()
    
    
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    
    center = (vmin + vmax) / 2.0
    vertices -= center 

    extents = vmax - vmin
    max_extent = np.max(extents) 
    
    if max_extent > 0:
        scale = target_size / max_extent
        vertices *= scale
    else:
        scale = 1.0

    
    num_tris = len(indices) // 3
    print(f"[load_obj_simple] {path}: verts={len(vertices)}, tris={num_tris}")
    print(f"[DEBUG] bbox min/max (após escala) = {vertices.min(axis=0)}, {vertices.max(axis=0)}")
    print(f"[DEBUG] scale usado: {scale}")

    return vertices, indices


def create_perspective_matrix(fov_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
  
    if near == far or near <= 0 or far <= 0 or fov_rad <= 0:
        raise ValueError("Parâmetros de projeção inválidos. 'near', 'far', e 'fov' devem ser positivos, e 'near' != 'far'.")

    tan_half_fov = math.tan(fov_rad / 2.0)
    if tan_half_fov == 0:
        raise ValueError("FOV inválido, tan(fov/2) é zero.")
        
    f = 1.0 / tan_half_fov
    
    range_inv = 1.0 / (near - far)
    
    A = f / aspect
    B = f
    
    
    C = (far + near) * range_inv
    D = (2.0 * far * near) * range_inv 

    mat = np.array([
        [A, 0.0, 0.0, 0.0],
        [0.0, B, 0.0, 0.0],
        [0.0, 0.0, C, D],
        [0.0, 0.0, -1.0, 0.0], 
    ], dtype=np.float32)
    
    return mat


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Vetor nulo não pode ser normalizado na função look_at.")
    return v / norm

def create_view_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    
    f = center - eye
    f = _normalize_vector(f) 
    
    
    u_norm = _normalize_vector(up) 
    
    s = np.cross(f, u_norm)
    s = _normalize_vector(s) 
    
    
    u = np.cross(s, f) 
    
    
    M = np.identity(4, dtype=np.float32)

    
    M[0, :3] = s  
    M[1, :3] = u  
    M[2, :3] = -f 


    M[:3, 3] = -np.dot(M[:3, :3], eye)

    return M


cube_vertices = np.array([
    -0.5, -0.5,  0.5,  
     0.5, -0.5,  0.5,  
     0.5,  0.5,  0.5,  
    -0.5,  0.5,  0.5,  
    
    -0.5, -0.5, -0.5,  
     0.5, -0.5, -0.5,  
     0.5,  0.5, -0.5,  
    -0.5,  0.5, -0.5,  
], dtype=np.float32)

cube_indices = np.array([
    0, 1, 2,    
    2, 3, 0,    
    
    1, 5, 6,    
    6, 2, 1,    
    
    5, 4, 7,    
    7, 6, 5,    
    
    4, 0, 3,    
    3, 7, 4,    
    
    3, 2, 6,    
    6, 7, 3,    
    
    4, 5, 1,    
    1, 0, 4     
], dtype=np.uint32)


TRACK_Y_LEVEL = -3.0 


track_vertices = np.array([
    # X     Y           Z
    -3.0, TRACK_Y_LEVEL,   5.0,  
     3.0, TRACK_Y_LEVEL,   5.0,  
     3.0, TRACK_Y_LEVEL, -200.0, 
    -3.0, TRACK_Y_LEVEL, -200.0, 
], dtype=np.float32)

track_indices = np.array([
    0, 1, 2,  
    2, 3, 0   
], dtype=np.uint32)

import numpy as np


def create_ground_plane(width: float, depth: float, y_level: float) -> tuple[np.ndarray, np.ndarray]:
    
    half_w = width / 2.0
    half_d = depth / 2.0

    vertices = np.array([
  
        -half_w, y_level,  half_d,  
         half_w, y_level,  half_d, 
         half_w, y_level, -half_d,  
        -half_w, y_level, -half_d, 
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2,  # T1
        2, 3, 0   # T2
    ], dtype=np.uint32)
    
    return vertices, indices



def create_custom_plane(x_min, x_max, z_min, z_max, y_level) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([
        [x_min, y_level, z_max],   
        [x_max, y_level, z_max],   
        [x_max, y_level, z_min],  
        [x_min, y_level, z_min],   
    ], dtype=np.float32).flatten() 

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
    return vertices, indices


LANE_POSITIONS = [-2.0, 0.0, 2.0] 

def initialize_game_state() -> Dict[str, Any]:
 
    INITIAL_LANE_INDEX = 1 
    
    game_state = {
        "lane_positions": LANE_POSITIONS,   
        "current_lane_index": INITIAL_LANE_INDEX, 
        
        "obstacles": [],                   
        
        "spawn_timer": 0.0,                

        "last_left_input": False,          
        "last_right_input": False,         
        
    }
    return game_state



def create_model_matrix(x: float, y: float, z: float, scale: float, rotation_y: float = 0.0) -> np.ndarray:
   
    model = np.identity(4, dtype=np.float32)

    
    model[0, 0] *= scale
    model[1, 1] *= scale
    model[2, 2] *= scale

    c = math.cos(rotation_y)
    s = math.sin(rotation_y)
    
    R_y = np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)

    
    model = np.dot(R_y, model) 

   
    model[0, 3] = x
    model[1, 3] = y
    model[2, 3] = z
    
    return model

def render_mesh(shader_id: int, vao_id: int, index_count: int, 
                model_matrix: np.ndarray, color: Tuple[float, float, float], 
                view_matrix: np.ndarray, proj_matrix: np.ndarray):
   
    gl.glUseProgram(shader_id)

   
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "model"), 
                          1, gl.GL_FALSE, model_matrix.T)
                          
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "view"), 
                          1, gl.GL_FALSE, view_matrix.T)
                          
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_id, "projection"), 
                          1, gl.GL_FALSE, proj_matrix.T)

    gl.glUniform3f(gl.glGetUniformLocation(shader_id, "objColor"), *color)
    
    gl.glBindVertexArray(vao_id)
    gl.glDrawElements(gl.GL_TRIANGLES, index_count, gl.GL_UNSIGNED_INT, None)
    gl.glBindVertexArray(0) # Desvincula o VAO


def setup_mesh_buffers(vertices: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
    

    if indices.size == 0 or vertices.size == 0:
        print(f"[ERROR] Mesh Buffer Setup Falhou: Vertices ou Índices vazios.")
        return 0, 0 
    
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, None)

    gl.glBindVertexArray(0)
    return vao, len(indices)

def handle_input(win: glfw._GLFWwindow, game_state: Dict[str, Any]):
    
    current_lane = game_state["current_lane_index"]
    last_left = game_state["last_left_input"]
    last_right = game_state["last_right_input"]
    
    left = glfw.get_key(win, glfw.KEY_LEFT) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_A) == glfw.PRESS
    right = glfw.get_key(win, glfw.KEY_RIGHT) == glfw.PRESS or \
            glfw.get_key(win, glfw.KEY_D) == glfw.PRESS

    if left and not last_left:  
        current_lane = max(0, current_lane - 1)
    if right and not last_right:
        current_lane = min(2, current_lane + 1)
    
    game_state["current_lane_index"] = current_lane
    game_state["last_left_input"] = left
    game_state["last_right_input"] = right

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
                model_matrix=player_model_mat,         
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
                    model_matrix=obs_model_mat,          
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