#include <stdio.h>

#include "utility.h"

using namespace af;
using namespace std;

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
                                    "ArrayFire CUDA OpenGL Interop", NULL, NULL);
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

        bool onscreen = true;           //Change to false for offscreen renderering
        init_glfw(width, height, channels, onscreen);

        // Initialize CUDA-OpenGL Data
        // Vertex Buffer
        GLuint vertex_b = 0;
        cudaGraphicsResource_t vertex_cuda;
        glGenBuffers(1, &vertex_b);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_b);
        glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        CUDA(cudaGraphicsGLRegisterBuffer(&vertex_cuda, vertex_b, cudaGraphicsRegisterFlagsNone));
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Index Buffer
        GLuint index_b = 0;
        cudaGraphicsResource_t index_cuda;
        glGenBuffers(1, &index_b);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_b);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(unsigned), indices, GL_STATIC_READ);   //One time copy
        CUDA(cudaGraphicsGLRegisterBuffer(&index_cuda, index_b, cudaGraphicsRegisterFlagsNone));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // Color Buffer
        GLuint color_b = 0;
        cudaGraphicsResource_t color_cuda;
        glGenBuffers(1, &color_b);
        glBindBuffer(GL_ARRAY_BUFFER, color_b);
        glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        CUDA(cudaGraphicsGLRegisterBuffer(&color_cuda, color_b, cudaGraphicsRegisterFlagsNone));
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Projection Buffer (Transform Feedback)
        GLuint project_b = 0;
        GLuint transform_b = 0;
        cudaGraphicsResource_t project_cuda;
        glGenBuffers(1, &project_b);
        glBindBuffer(GL_ARRAY_BUFFER, project_b);
        glBufferData(GL_ARRAY_BUFFER, 3 * num_vertices * sizeof(float), NULL, GL_DYNAMIC_DRAW);
        CUDA(cudaGraphicsGLRegisterBuffer(&project_cuda, project_b, cudaGraphicsRegisterFlagsNone));
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Render Buffer
        GLuint render_b = 0;
        cudaGraphicsResource_t render_cuda;
        glGenRenderbuffers(1, &render_b);
        glBindRenderbuffer(GL_RENDERBUFFER, render_b);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
        CUDA(cudaGraphicsGLRegisterImage(&render_cuda, render_b, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsNone));
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        // Depth Buffer - for off screen rendering
        GLuint depth_b = 0;
        glGenRenderbuffers(1, &depth_b);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_b);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

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

            //Unlock arrays
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
                glfwMakeContextCurrent(NULL);       // Need to unset context sinze ArrayFire uses its own for graphics
                image(af_image);
                saveimage("test.png", af_image);
                getchar();
                glfwMakeContextCurrent(window);
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
