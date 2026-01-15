#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texCoord;
layout(location = 3) in vec3 a_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_color;
out vec2 v_texCoord;
out vec3 v_fragPos;
out vec3 v_normal;

void main()
{
    vec4 worldPos = model * vec4(a_position, 1.0);
    v_fragPos = worldPos.xyz;

    v_normal = mat3(transpose(inverse(model))) * a_normal;

    v_color = a_color;
    v_texCoord = a_texCoord;

    gl_Position = projection * view * worldPos;
}
