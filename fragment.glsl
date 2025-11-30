#version 330 core

// Cor de saída (Display)
out vec4 FragColor;

// Entrada do Vertex Shader
in vec2 TexCoord; // Coordenada UV

// Uniforms
uniform vec3 objColor; // Cor sólida (usada se não houver textura)

// NOVO: Sampler para a textura
uniform sampler2D textureSampler;

// NOVO: Flag para decidir entre cor sólida (0) ou textura (1)
uniform int useTexture; 

void main()
{
    if (useTexture == 1) {
        // Usa a cor da textura (amostrada na coordenada TexCoord)
        // O quarto componente (w/alpha) é importante para transparência futura
        FragColor = texture(textureSampler, TexCoord); 
    } else {
        // Usa a cor sólida (para a pista e obstáculos)
        FragColor = vec4(objColor, 1.0);
    }
}