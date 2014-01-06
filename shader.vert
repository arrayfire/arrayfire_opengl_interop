#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Color;

out vec3 position;
out vec3 color;
void main()
{
    position = vec3(Position.xyz / 2.0); //to make sure projection is working
    color = Color;
    gl_Position = vec4(Position, 1.0);
}
