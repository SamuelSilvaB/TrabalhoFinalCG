#version 330 core

// Attrib 0: Posição XYZ
layout (location = 0) in vec3 aPos;

// Attrib 1: Coordenadas de Textura UV (NOVO)
layout (location = 1) in vec2 aTexCoord; 

// Saída para o Fragment Shader (NOVO)
out vec2 TexCoord;

// Uniforms (Matrizes)
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Passa a coordenada de textura para o fragment shader
    TexCoord = aTexCoord; 
    
    // Calcula a Posição final (MVP)
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}