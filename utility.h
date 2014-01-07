#ifndef UTILITY
#define UTILITY

#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <arrayfire.h>
#include <af/utils.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "util_shaders.hpp"

#define CUDA(x) do {                                                        \
    cudaError_t err = (x);                                                  \
    if(cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA Error in %s:%d: %s \nReturned: %s.\n",        \
                __FILE__, __LINE__, #x, cudaGetErrorString(err) );          \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while(0)

static bool display;
static GLFWwindow* window = NULL;
static GLuint shader_program = 0;

static const float vertices[] = {
    0.0f,  0.8f, 0.0f,
   -0.6f, -0.4f, 0.0f,
    0.6f, -0.4f, 0.0f,
};

static const float colors[] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
};

static const unsigned indices[] = {
    0, 1, 2
};


static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static void key_callback(GLFWwindow* wind, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(wind, GL_TRUE);
    }
}

af::array unpacked(const af::array& a)
{
    return af::flip(af::reorder(a, 2, 1, 0), 0);
}

af::array packed(const af::array& a)
{
    return af::reorder(a, 2, 1, 0);
}

void
init_glfw(const int buffer_width, const int buffer_height,
          const int buffer_depth, const bool set_display)
{
    if (window == NULL) {
        glfwSetErrorCallback(error_callback);
        if (!glfwInit()) {
            std::cerr << "ERROR: GLFW wasn't able to initalize" << std::endl;
            exit(EXIT_FAILURE);
        }

        if(buffer_depth <=0 || buffer_depth > 4) {
            std::cerr << "ERROR: Depth value must be between 1 and 4" << std::endl;
        }

        display = set_display;
        if(!display)
            glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

        glfwWindowHint(GLFW_DEPTH_BITS, buffer_depth * 8);
        glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
        window = glfwCreateWindow(buffer_width, buffer_height,
                                    "ArrayFire OpenGL Interop", NULL, NULL);
        if (!window) {
            glfwTerminate();
            //Comment/Uncomment these lines incase using fall backs
            //return;
            std::cerr << "ERROR: GLFW couldn't create a window." << std::endl;
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        int b_width = buffer_width;
        int b_height = buffer_height;
        int b_depth = buffer_depth;
        glfwGetFramebufferSize(window, &b_width, &b_height);
        glfwSetTime(0.0);

        glfwSetKeyCallback(window, key_callback);

        //GLEW Initialization - Must be done
        GLenum res = glewInit();
        if (res != GLEW_OK) {
            std::cerr << "Error Initializing GLEW | Exiting" << std::endl;
            exit(-1);
        }
        //Put in resize
        glViewport(0, 0, b_width, b_height);
    }
}

void
init_program(const char* vertex_shader_path, const char* frag_shader_path)
{
    shaders_t shaders = loadShaders(vertex_shader_path, frag_shader_path);

    shader_program = glCreateProgram();

    attachAndLinkProgram(shader_program, shaders);

    glUseProgram(shader_program);
}

void
init_projection(GLuint& transform_b, GLuint& project_b)
{
    glGenTransformFeedbacks(1, &transform_b);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transform_b);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, project_b);
    const GLchar* shader_variables[] = {"position"};
    glTransformFeedbackVaryings(shader_program, 1, shader_variables, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(shader_program);

    GLint success;
    GLchar log[1024] = { 0 };
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (success == 0) {
        glGetProgramInfoLog(shader_program, sizeof(log), NULL, log);
        std::cerr << "Error linking shader program:" << log << std::endl;
        exit(-1);
    }

    glUseProgram(shader_program);
    glEnableVertexAttribArray(0);
}

void
bind_framebuffer(GLuint& image, GLuint& depth, GLuint& frame_buffer)
{
    if (image != 0 && depth != 0) {
        glGenFramebuffers(1, &frame_buffer);
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);

        glBindRenderbuffer(GL_RENDERBUFFER, image);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, image);

        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        switch (status) {
        case GL_FRAMEBUFFER_UNDEFINED:
            std::cerr << "FBO Undefined\n" << std::endl;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT :
            std::cerr << "FBO Incomplete Attachment\n" << std::endl;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT :
            std::cerr << "FBO Missing Attachment\n" << std::endl;
            break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER :
            std::cerr << "FBO Incomplete Draw Buffer\n" << std::endl;
            break;
        case GL_FRAMEBUFFER_UNSUPPORTED :
            std::cerr << "FBO Unsupported\n" << std::endl;
            break;
        case GL_FRAMEBUFFER_COMPLETE:
            //std::cerr << "FBO OK\n" << std::endl;
            break;
        default:
            std::cerr << "FBO Problem?\n" << std::endl;
        }

        if (status != GL_FRAMEBUFFER_COMPLETE) {
            frame_buffer = 0;
            exit(-1);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

af::array
rotate_z(const af::array& input, const float theta)
{
    float rot_mat[] = { cos(theta),  sin(theta), 0,
                       -sin(theta),  cos(theta), 0,
                        0,           0,          1};
    af::array rot = af::array(3, 3, rot_mat);
    af::array output = input.copy();
    gfor(af::array i, input.dims(1)) {
        output(af::span, i) = af::matmul(rot, input(af::span, i));
    }
    return output;
}

#endif // UTILITY
