# version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;
out vec2 v_texCoord;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
    v_texCoord = a_texCoord;
}