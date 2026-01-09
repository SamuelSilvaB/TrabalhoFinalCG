#version 330 core

in vec3 v_color;
in vec2 v_texCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform bool useTexture;
uniform float time;

void main()
{
    vec2 uv = v_texCoord;

    // movimento suave da água
    uv.y += time * 0.05;
    uv.x += sin(time + uv.y * 10.0) * 0.02;

    if (useTexture)
        FragColor = texture(texture1, uv) * vec4(v_color, 1.0);
    else
        FragColor = vec4(v_color, 1.0);
}
