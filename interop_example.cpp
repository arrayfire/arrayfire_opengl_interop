#include <stdio.h>

#include "utility.h"

using namespace af;
using namespace std;

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
render(GLuint vertex_b,
       GLuint color_b,
       GLuint index_b,
       GLuint transform_b,
       unsigned num_triangles,
       unsigned num_vertices,
       GLuint frame_buffer = 0)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_b);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, color_b);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_b);
    glDisableClientState(GL_VERTEX_ARRAY);

    glDrawElements(GL_TRIANGLES, num_triangles * 3, GL_UNSIGNED_INT, 0);

    // Transform Feedback rendering
    // Gets values back from vertex shader - optional
    glEnable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transform_b);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, num_vertices);
    glEndTransformFeedback();
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDisable(GL_RASTERIZER_DISCARD);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glfwSwapBuffers(window);

    glfwPollEvents();
}

void
render_to_framebuffer(GLuint vertex_b,
                      GLuint color_b,
                      GLuint index_b,
                      GLuint transform_b,
                      unsigned num_triangles,
                      unsigned num_vertices,
                      GLuint frame_buffer)
{
    // Render to generated frambuffer
    // Used to copy data back into ArrayFire
    if(!display && frame_buffer != 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
        GLuint attachment = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &attachment);

        render(vertex_b, color_b, index_b, transform_b,
            num_triangles, num_vertices, frame_buffer);
    } else {
        std::cerr << "Framebuffer not valid" << std::endl;
    }
}

void
render_to_screen(GLuint vertex_b,
                 GLuint color_b,
                 GLuint index_b,
                 GLuint transform_b,
                 unsigned num_triangles,
                 unsigned num_vertices)
{
    render(vertex_b, color_b, index_b, transform_b,
            num_triangles, num_vertices);
}

void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              cudaGraphicsResource_t* cuda_resource,
              const unsigned size,
              GLenum buffer_usage,
              const void* data = NULL)
{
    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);
    CUDA(cudaGraphicsGLRegisterBuffer(cuda_resource, buffer, cudaGraphicsRegisterFlagsNone));
    glBindBuffer(buffer_target, 0);
}

void
create_buffer(GLuint& buffer,
              GLenum format,
              const unsigned width,
              const unsigned height,
              cudaGraphicsResource_t* cuda_resource = NULL)
{
    glGenRenderbuffers(1, &buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
    if(cuda_resource != NULL)
        CUDA(cudaGraphicsGLRegisterImage(cuda_resource, buffer, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsNone));
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
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

int main(int argc, char* argv[])
{
    try {
        int device = argc > 1? atoi(argv[1]):0;
        af::deviceset(device);
        af::info();

        int width = 400, height = 400, channels = 4;
        int num_triangles = 1, num_vertices = 3;

        array af_indices = array(3, indices);
        array af_vertices = array(3, num_vertices, vertices);   //3 floats per vertex, 3 vertices
        array af_colors = array(3, num_vertices, colors);       //3 floats per color, 3 colors
        array af_image = constant(0.0, height, width, channels);
        array pkd_image = constant(0.0, channels, width, height);
        array af_project = constant(-1.0, 3, num_vertices);

        bool onscreen = false;           //Change to false for offscreen renderering
        init_glfw(width, height, channels, onscreen);

        // Initialize CUDA-OpenGL Data
        // Vertex Buffer
        GLuint vertex_b = 0;
        cudaGraphicsResource_t vertex_cuda;
        create_buffer(vertex_b, GL_ARRAY_BUFFER, &vertex_cuda,
                      3 * num_vertices * sizeof(float), GL_DYNAMIC_DRAW);

        // Index Buffer
        GLuint index_b = 0;
        cudaGraphicsResource_t index_cuda;
        create_buffer(index_b, GL_ELEMENT_ARRAY_BUFFER, &index_cuda,
                      3 * sizeof(unsigned), GL_STATIC_READ, indices);

        // Color Buffer
        GLuint color_b = 0;
        cudaGraphicsResource_t color_cuda;
        create_buffer(color_b, GL_ARRAY_BUFFER, &color_cuda,
                      3 * num_vertices * sizeof(float), GL_DYNAMIC_DRAW);

        // Projection Buffer (Transform Feedback)
        GLuint project_b = 0;
        GLuint transform_b = 0;
        cudaGraphicsResource_t project_cuda;
        create_buffer(project_b, GL_ARRAY_BUFFER, &project_cuda,
                      3 * num_vertices * sizeof(float), GL_DYNAMIC_DRAW);

        // Render Buffer
        GLuint render_b = 0;
        cudaGraphicsResource_t render_cuda;
        create_buffer(render_b, GL_RGBA32F, width, height, &render_cuda);

        // Depth Buffer - for off screen rendering
        GLuint depth_b = 0;
        create_buffer(depth_b, GL_DEPTH_COMPONENT, width, height);

        //Required for framebuffer copy
        GLuint frame_b = 0;
        bind_framebuffer(render_b, depth_b, frame_b);

        // Initialize shaders
        init_program("shader.vert", "shader.frag");

        // Initialize transform feedback
        init_projection(transform_b, project_b);

        while(!glfwWindowShouldClose(window)) {
            af_vertices = rotate_z(af_vertices, af::Pi / 180.0);
            // Copy vertices to OpenGL
            float* d_vertices = af_vertices.device<float>();
            copy_from_device_pointer(vertex_cuda,
                                     d_vertices,
                                     GL_ARRAY_BUFFER,
                                     3 * num_vertices * sizeof(float));

            // Copy colors to OpenGL
            float* d_colors = af_colors.device<float>();
            copy_from_device_pointer(color_cuda,
                                     d_colors,
                                     GL_ARRAY_BUFFER,
                                     3 * num_vertices * sizeof(float));

            // Render
            if(onscreen)
                render_to_screen(vertex_b, color_b, index_b, transform_b,
                                 num_triangles, num_vertices);
            else
                render_to_framebuffer(vertex_b, color_b, index_b, transform_b,
                                      num_triangles, num_vertices, frame_b);

            // Copy transform feedback
            float* d_project = af_project.device<float>();
            copy_to_device_pointer(project_cuda,
                                   d_project,
                                   GL_ARRAY_BUFFER,
                                   3 * num_vertices * sizeof(float));

            // Unlock arrays
            af_vertices.unlock();
            af_colors.unlock();
            af_project.unlock();

            // If off screen renderering, copy data back to array
            if(!onscreen) {
                float* d_image = pkd_image.device<float>();
                copy_to_device_pointer(render_cuda,
                                       d_image,
                                       GL_RENDERBUFFER,
                                       width * height * channels * sizeof(float));
                pkd_image.unlock();
                af_image = unpacked(pkd_image);
                saveimage("test.png", af_image);
                printf("Image saved\n");
                break;
            }
        }

        //Cleanup
        delete_buffer(vertex_b, GL_ARRAY_BUFFER, vertex_cuda);
        delete_buffer(color_b, GL_ARRAY_BUFFER, color_cuda);
        delete_buffer(index_b, GL_ELEMENT_ARRAY_BUFFER, index_cuda);
        delete_buffer(project_b, GL_ARRAY_BUFFER, project_cuda);
        delete_buffer(render_b, GL_RENDERBUFFER, render_cuda);

        glBindRenderbuffer(GL_RENDERBUFFER, depth_b);
        glDeleteRenderbuffers(1, &depth_b);
        depth_b = 0;
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
