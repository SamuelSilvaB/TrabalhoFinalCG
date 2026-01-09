#version 330 core

in vec3 v_color;
in vec2 v_texCoord;
out vec4 FragColor;

uniform sampler2D texture1;
uniform bool useTexture;

void main()
{
    if (useTexture)
        FragColor = texture(texture1, v_texCoord) * vec4(v_color, 1.0);
    else
        FragColor = vec4(v_color, 1.0);
}