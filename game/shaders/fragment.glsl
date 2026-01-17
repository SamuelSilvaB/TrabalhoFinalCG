#version 330 core

in vec3 v_fragPos;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_texCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform bool useTexture;
uniform bool isWater;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

uniform float time;

void main()
{
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(lightPos - v_fragPos);

    // -------- AMBIENT --------
    float ambientStrength = isWater ? 0.3 : 0.15;
    vec3 ambient = ambientStrength * lightColor;

    // -------- DIFFUSE --------
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // -------- SPECULAR --------
    vec3 viewDir = normalize(viewPos - v_fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);

    float specStrength = isWater ? 0.8 : 0.4;
    float shininess   = isWater ? 64.0 : 32.0;

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specStrength * spec * lightColor;

    // -------- COR BASE --------
    vec2 uv = v_texCoord;
    vec3 baseColor;

    if (isWater) {
        uv.x += time * 0.05;
        uv.y += sin(time * 0.5) * 0.02;
    }

    if (useTexture)
        baseColor = texture(texture1, uv).rgb;
    else
        baseColor = v_color;


    vec3 result = (ambient + diffuse + specular) * baseColor;
    FragColor = vec4(result, 1.0);
}
