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

static float vertices[] = {
    0.0f,  0.8f, 0.0f,
   -0.6f, -0.8f, 0.0f,
    0.6f, -0.8f, 0.0f,
};

static float colors[] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
};

static unsigned indices[] = {
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
init_program(const char* vertex_shader_path, const char* frag_shader_path)
{
    shaders_t shaders = loadShaders(vertex_shader_path, frag_shader_path);

    shader_program = glCreateProgram();

    attachAndLinkProgram(shader_program, shaders);

    glUseProgram(shader_program);
}

void
unmap_resource(cudaGraphicsResource_t cuda_resource,
               bool is_mapped)
{
    if (is_mapped) {
        CUDA(cudaGraphicsUnmapResources(1, &cuda_resource));
        is_mapped = false;
    }
}

// Gets the device pointer from the mapped resource
// Sets is_mapped to true
template<typename T>
void copy_from_device_pointer(cudaGraphicsResource_t cuda_resource,
                              T& d_ptr,
                              GLuint buffer_target,
                              const unsigned size)
{
    CUDA(cudaGraphicsMapResources(1, &cuda_resource));
    bool is_mapped = true;
    if (buffer_target == GL_RENDERBUFFER) {
        cudaArray* array_ptr = NULL;
        CUDA(cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_resource, 0, 0));
        CUDA(cudaMemcpyToArray(array_ptr, 0, 0, d_ptr, size, cudaMemcpyDeviceToDevice));
    } else {
        T* opengl_ptr = NULL;
        CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size, cuda_resource));
        CUDA(cudaMemcpy(opengl_ptr, d_ptr, size, cudaMemcpyDeviceToDevice));
    }
    unmap_resource(cuda_resource, is_mapped);
}

// Gets the device pointer from the mapped resource
// Sets is_mapped to true
template<typename T>
void copy_to_device_pointer(cudaGraphicsResource_t cuda_resource,
                            T& d_ptr,
                            GLuint buffer_target,
                            const unsigned size)
{
    cudaGraphicsMapResources(1, &cuda_resource);
    bool is_mapped = true;
    if (GL_RENDERBUFFER == buffer_target) {
        cudaArray* array_ptr;
        CUDA(cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_resource, 0, 0));
        CUDA(cudaMemcpyFromArray(d_ptr, array_ptr, 0, 0, size, cudaMemcpyDeviceToDevice));
    } else {
        T* opengl_ptr = NULL;
        CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size, cuda_resource));
        CUDA(cudaMemcpy(d_ptr, opengl_ptr, size, cudaMemcpyDeviceToDevice));
    }
    unmap_resource(cuda_resource, is_mapped);
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

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              cudaGraphicsResource_t cuda_resource)
{
    CUDA(cudaGraphicsUnregisterResource(cuda_resource));
    if (buffer_target == GL_RENDERBUFFER) {
        glBindRenderbuffer(buffer_target, buffer);
        glDeleteRenderbuffers(1, &buffer);
        buffer = 0;
    } else {
        glBindBuffer(buffer_target, buffer);
        glDeleteRenderbuffers(1, &buffer);
        buffer = 0;
    }
}

#endif // UTILITY
