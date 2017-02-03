#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <stack>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
using namespace std;

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>
#include "common/tga.h"
#include "common/tga.c"
#include "common/ray_casting.h"

const int window_width = 600, window_height = 600;

typedef struct Vertex {
    float Position[4];
    float Color[4];
    float Normal[3];
    float TexCoord[2];
    void SetPosition(float *coords) {
        Position[0] = coords[0];
        Position[1] = coords[1];
        Position[2] = coords[2];
        Position[3] = 1.0;
    }
    void SetColor(float *color) {
        Color[0] = color[0];
        Color[1] = color[1];
        Color[2] = color[2];
        Color[3] = color[3];
    }
    void SetNormal(float *coords) {
        Normal[0] = coords[0];
        Normal[1] = coords[1];
        Normal[2] = coords[2];
    }
    void SetTexCoord(float *coords) {
        TexCoord[0] = coords[0];
        TexCoord[1] = coords[1];
    }
};

typedef struct point {
    float x, y, z;
    point(const float x = 0, const float y = 0, const float z = 0) : x(x), y(y), z(z){};
    point(float *coords) : x(coords[0]), y(coords[1]), z(coords[2]){};
    point operator -(const point& a)const {
        return point(x - a.x, y - a.y, z - a.z);
    }
    point operator +(const point& a)const {
        return point(x + a.x, y + a.y, z + a.z);
    }
    point operator *(const float& a)const {
        return point(x*a, y*a, z*a);
    }
    point operator /(const float& a)const {
        return point(x / a, y / a, z / a);
    }
    float* toArray() {
        float array[] = { x, y, z, 1.0f };
        return array;
    }
};

#define Axes 0
#define Grid 1
#define Face 2
#define CtrlMesh 3

#define up_left     (i-old_side-1)
#define up          (i-old_side)
#define up_right    (i-old_side+1)
#define left        (i-1)
#define center      (i)
#define right       (i+1)
#define down_left   (i+old_side-1)
#define down        (i+old_side)
#define down_right  (i+old_side+1)

// function prototypes
int initWindow(void);
void initOpenGL(void);
void loadObject(char*, glm::vec4, Vertex * &, GLushort* &, int);
void createVAOs(Vertex[], GLushort[], int);
void createObjects(void);
void pickObject(void);
void renderScene(void);
void cleanup(void);
static void keyCallback(GLFWwindow*, int, int, int, int);
static void mouseCallback(GLFWwindow*, int, int, int);
static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void cameraRotation(int key, double scroll_offset);
void automateFitting_z();
void automateFitting();
bool CCW_test(float *p, float *q, float *r);
void create_original_CtrlMesh();
void subdivision();
void exportPoints();
void importPoints();
void switchProfile(int profile);
void smileAnimation();
void upsetAnimation();
void frownAnimation();

// GLOBAL VARIABLES
GLFWwindow* window;

glm::mat4 gProjectionMatrix;
glm::mat4 gViewMatrix;

GLuint programID;
GLuint pickingProgramID;
GLuint textrueProgramID;

const GLuint NumObjects = 4;	// ATTN: THIS NEEDS TO CHANGE AS YOU ADD NEW OBJECTS
GLuint VertexArrayId[NumObjects] = { 0 };
GLuint VertexBufferId[NumObjects] = { 0 };
GLuint IndexBufferId[NumObjects] = { 0 };

size_t NumIndices[NumObjects] = { 0 };
size_t VertexBufferSize[NumObjects] = { 0 };
size_t IndexBufferSize[NumObjects] = { 0 };

Vertex* FaceVerts;
GLushort* FaceIdcs;

GLuint ModelMatrixID;
GLuint ViewMatrixID;
GLuint ProjMatrixID;
GLuint PickingModelMatrixID;
GLuint PickingViewMatrixID;
GLuint PickingProjMatrixID;
GLuint ObjectColorID;
GLuint LightID;
GLuint TextureModelMatrixID;
GLuint TextureViewMatrixID;
GLuint TextureProjMatrixID;
GLuint TextureID;
GLuint TexCoordID;
GLuint gPickedIndex = -1;
GLuint dragIndex = -1;

const int initialMeshSide = 40;
const float camera_rotate_speed = 5.0;
const vec3 cameraPos = vec3(20.0, 20.0, 20.0);
const vec3 cameraLookat = vec3(0.0, 10.0, 0.0);
int cameraRedDegree = 36;
const float light_pos[3] = {0.0, 15.0, 30.0};
float PointSize = 5.0;
bool showControlMesh = true;
bool showTexture = true;
bool showFace = true;
bool lastFrameDrag = false;
bool shiftPressed = false;
bool showSubdivision = false;
bool animate_smile = false;
bool animate_upset = false;
bool animate_frown = false;
float ctrlMesh_x_offset = 0.0;
float ctrlMesh_y_offset = 0.0;
float texture_s_startPoint = 0.0;
float texture_s_endPoint = 0.0;
float texture_t_startPoint = 0.0;
float texture_t_endPoint = 0.0;
float left_mouth_corner_x = 0.0;
float left_mouth_corner_y = 0.0;
float right_mouth_corner_x = 0.0;
float right_mouth_corner_y = 0.0;
float left_eyebrow_center_x = 0.0;
float left_eyebrow_center_y = 0.0;
float right_eyebrow_center_x = 0.0;
float right_eyebrow_center_y = 0.0;

Vertex *CtrlMeshVerticesVerts = NULL;
GLushort *CtrlMeshVerticesIdcs_horizontal = NULL;
GLushort *CtrlMeshVerticesIdcs_vertical = NULL;
GLushort *CtrlMeshVerticesIdcs_texture = NULL;
GLushort CtrlMeshTotalPoints = 0;
GLushort CtrlMeshSide = 0;

glm::mat4 ModelMatrix[NumObjects];
glm::mat4 CamModelMatrix;
glm::mat4 CamTranslationMatrix;
glm::mat4 CamRotationMatrix;

void loadObject(char* file, glm::vec4 color, Vertex * &out_Vertices, GLushort* &out_Indices, int ObjectId) {
    // Read our .obj file
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    bool res = loadOBJ(file, vertices, normals);
    
    std::vector<GLushort> indices;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;
    indexVBO(vertices, normals, indices, indexed_vertices, indexed_normals);
    
    const size_t vertCount = indexed_vertices.size();
    const size_t idxCount = indices.size();
    
    // populate output arrays
    out_Vertices = new Vertex[vertCount];
    for (int i = 0; i < vertCount; i++) {
        out_Vertices[i].SetPosition(&indexed_vertices[i].x);
        out_Vertices[i].SetNormal(&indexed_normals[i].x);
        out_Vertices[i].SetColor(&color[0]);
    }
    out_Indices = new GLushort[idxCount];
    for (int i = 0; i < idxCount; i++) {
        out_Indices[i] = indices[i];
    }
    
    // set global variables!!
    NumIndices[ObjectId] = idxCount;
    VertexBufferSize[ObjectId] = sizeof(out_Vertices[0]) * vertCount;
    IndexBufferSize[ObjectId] = sizeof(GLushort) * idxCount;
}


void createObjects(void) {
    //-- COORDINATE AXES --//
    Vertex CoordVerts[] =
    {
        { { 0.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
        { { 10.0, 0.0, 0.0, 1.0 }, { 1.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
        { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
        { { 0.0, 10.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
        { { 0.0, 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
        { { 0.0, 0.0, 10.0, 1.0 }, { 0.0, 0.0, 1.0, 1.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0 } },
    };
    
    VertexBufferSize[Axes] = sizeof(CoordVerts);	// ATTN: this needs to be done for each hand-made object with the ObjectID (subscript)
    createVAOs(CoordVerts, NULL, Axes);
    
    //-- GRID --//
    Vertex GridVerts[84];
    
    float tempGridLeft_X[4] = {-10.0, 0.0, -10.0, 1.0};
    float tempGridRight_X[4] = {-10.0, 0.0, 10.0, 1.0};
    for(int i = 0; i < 42; i += 2) {
        GridVerts[i].SetPosition(tempGridLeft_X);
        GridVerts[i+1].SetPosition(tempGridRight_X);
        tempGridLeft_X[0]++;
        tempGridRight_X[0]++;
    }
    float tempGridLeft_Z[4] = {10.0, 0.0, -10.0, 1.0};
    float tempGridRight_Z[4] = {-10.0, 0.0, -10.0, 1.0};
    for(int i = 42; i < 84; i += 2) {
        GridVerts[i].SetPosition(tempGridLeft_Z);
        GridVerts[i+1].SetPosition(tempGridRight_Z);
        tempGridLeft_Z[2]++;
        tempGridRight_Z[2]++;
    }
    
    float GridColor[4] = {1.0, 1.0, 1.0, 1.0};
    float GridNormal[4] = {1.0, 1.0, 1.0, 1.0};
    float GridTexCoord[2] = {0.0, 0.0};
    for(int i = 0; i < 84; i++) {
        GridVerts[i].SetColor(GridColor);
        GridVerts[i].SetNormal(GridNormal);
        GridVerts[i].SetTexCoord(GridTexCoord);
    }
    
    VertexBufferSize[Grid] = sizeof(GridVerts);
    createVAOs(GridVerts, NULL, Grid);
    
    //-- Control Mesh --//  //-- .OBJs --//  //-- Texture --//
    switchProfile(GLFW_KEY_1);
    
    // initialize ModelMatrix
    for(int i = 0; i < NumObjects; i++)
    ModelMatrix[i] = glm::mat4(1.0);
    
    /*mat4 faceTranslate = mat4(1.0);
     mat4 faceRotate = mat4(1.0);
     mat4 faceScale = mat4(1.0);
     
     faceTranslate = translate(faceTranslate, vec3(0.0f, -1.0f, 0.0f));
     faceRotate = faceRotate * toMat4(quat(angleAxis(3.3f, vec3(0.0, 1.0, 0.0))));
     faceScale = scale(faceScale, vec3(60.0f));
     
     ModelMatrix[Face] = faceTranslate * faceRotate * faceScale;*/
    
    CamModelMatrix = glm::mat4(1.0);
}

void renderScene(void){
    // Black background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    // Re-clear the screen for real rendering
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if(animate_smile)
    smileAnimation();
    if(animate_upset)
    upsetAnimation();
    if(animate_frown)
    frownAnimation();
    
    glUseProgram(programID);
    {
        static glm::vec3 lightPos = glm::vec3(light_pos[0], light_pos[1], light_pos[2]);
        glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);
        float lightDistance = distance(lightPos, vec3(0.0));
        GLuint lightDistanceID = glGetUniformLocation(programID, "light_distance");
        glUniform1f(lightDistanceID, lightDistance);
        
        glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
        glUniformMatrix4fv(ProjMatrixID, 1, GL_FALSE, &gProjectionMatrix[0][0]);
        
        // draw CoordAxes
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[Axes][0][0]);
        glBindVertexArray(VertexArrayId[Axes]);
        glDrawArrays(GL_LINES, 0, 6);
        
        // draw Grid
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[Grid][0][0]);
        glBindVertexArray(VertexArrayId[Grid]);
        glDrawArrays(GL_LINES, 0, 84);
        
        // draw Face
        if(showFace) {
            glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[Face][0][0]);
            glBindVertexArray(VertexArrayId[Face]);
            glDrawElements(GL_TRIANGLES, NumIndices[Face], GL_UNSIGNED_SHORT, (void*)0);
        }
        
        // draw Control mesh
        if(showControlMesh) {
            glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[CtrlMesh][0][0]);
            glBindVertexArray(VertexArrayId[CtrlMesh]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[CtrlMesh]);
            
            glBufferSubData(GL_ARRAY_BUFFER, 0, VertexBufferSize[CtrlMesh], CtrlMeshVerticesVerts);
            
            // Control mesh points
            if(!showSubdivision) {
                float CtrlMeshVerticesColor[4] = {0.0, 1.0, 0.0, 1.0};
                for(int i = 0; i < CtrlMeshTotalPoints; i++)
                CtrlMeshVerticesVerts[i].SetColor(CtrlMeshVerticesColor);
                glBufferSubData(GL_ARRAY_BUFFER, 0, VertexBufferSize[CtrlMesh], CtrlMeshVerticesVerts);
            }
            
            glDisable(GL_PROGRAM_POINT_SIZE);
            glPointSize(PointSize);
            glDrawArrays(GL_POINTS, 0, CtrlMeshTotalPoints);
            glEnable(GL_PROGRAM_POINT_SIZE);
            
            // Control mesh wireframe
            if(!showSubdivision) {
                float CtrlMeshGridColor[4] = {1.0, 1.0, 1.0, 1.0};
                for(int i = 0; i < CtrlMeshTotalPoints; i++)
                CtrlMeshVerticesVerts[i].SetColor(CtrlMeshGridColor);
                glBufferSubData(GL_ARRAY_BUFFER, 0, VertexBufferSize[CtrlMesh], CtrlMeshVerticesVerts);
            }
            
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, CtrlMeshTotalPoints * sizeof(GLushort), CtrlMeshVerticesIdcs_vertical);
            for(int i = 0; i < CtrlMeshSide; i++)
            glDrawElements(GL_LINE_STRIP, CtrlMeshSide, GL_UNSIGNED_SHORT, (void*)(i * (CtrlMeshSide) * sizeof(GLushort)));
            
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, CtrlMeshTotalPoints * sizeof(GLushort), CtrlMeshVerticesIdcs_horizontal);
            for(int i = 0; i < CtrlMeshSide; i++)
            glDrawElements(GL_LINE_STRIP, CtrlMeshSide, GL_UNSIGNED_SHORT, (void*)(i * (CtrlMeshSide) * sizeof(GLushort)));
        }
        
        glBindVertexArray(0);
    }
    glUseProgram(0);
    
    glUseProgram(textrueProgramID);
    {
        static glm::vec3 lightPos = glm::vec3(light_pos[0], light_pos[1], light_pos[2]);
        glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);
        float lightDistance = distance(lightPos, vec3(0.0));
        GLuint lightDistanceID = glGetUniformLocation(textrueProgramID, "light_distance");
        glUniform1f(lightDistanceID, lightDistance);
        
        glUniformMatrix4fv(TextureViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
        glUniformMatrix4fv(TextureProjMatrixID, 1, GL_FALSE, &gProjectionMatrix[0][0]);
        
        if(showTexture) {
            glUniformMatrix4fv(TextureModelMatrixID, 1, GL_FALSE, &ModelMatrix[CtrlMesh][0][0]);
            glBindVertexArray(VertexArrayId[CtrlMesh]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[CtrlMesh]);
            
            glBufferSubData(GL_ARRAY_BUFFER, 0, VertexBufferSize[CtrlMesh], CtrlMeshVerticesVerts);
            
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, IndexBufferSize[CtrlMesh], CtrlMeshVerticesIdcs_texture);
            glDisable(GL_CULL_FACE);
            glDrawElements(GL_TRIANGLES, IndexBufferSize[CtrlMesh] / sizeof(GLushort), GL_UNSIGNED_SHORT, (void*)0);
            glEnable(GL_CULL_FACE);
        }
        
        glBindVertexArray(0);
    }
    glUseProgram(0);
    
    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void pickObject(void) {
    float pickingColor[CtrlMeshTotalPoints][2];
    for(int i = 0; i < CtrlMeshTotalPoints; i++) {
        pickingColor[i][0] = float(i % 255) / 255.0f;
        pickingColor[i][1] = (i/255) / 255.0f;
    }
    
    // Clear the screen in white
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(pickingProgramID);
    {
        glUniformMatrix4fv(PickingViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
        glUniformMatrix4fv(PickingProjMatrixID, 1, GL_FALSE, &gProjectionMatrix[0][0]);
        
        // ATTN: DRAW YOUR PICKING SCENE HERE. REMEMBER TO SEND IN A DIFFERENT PICKING COLOR FOR EACH OBJECT BEFOREHAND
        
        // draw Control mesh
        if(showControlMesh) {
            glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[CtrlMesh][0][0]);
            glBindVertexArray(VertexArrayId[CtrlMesh]);
            glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[CtrlMesh]);
            glBufferSubData(GL_ARRAY_BUFFER, 0, VertexBufferSize[CtrlMesh], CtrlMeshVerticesVerts);
            
            // Control mesh points
            glDisable(GL_PROGRAM_POINT_SIZE);
            glPointSize(PointSize);
            for(int i = 0; i < CtrlMeshTotalPoints; i++) {
                glUniform1fv(ObjectColorID, 2, pickingColor[i]);
                glDrawArrays(GL_POINTS, i, 1);
            }
            glEnable(GL_PROGRAM_POINT_SIZE);
        }
        
        glBindVertexArray(0);
    }
    glUseProgram(0);
    
    // Wait until all the pending drawing commands are really done.
    glFlush();
    glFinish();
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    // Read the pixel at the center of the screen.
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    unsigned char data[4];
    glReadPixels(xpos, window_height - ypos, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data); // OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top
    
    // Convert the color back to an integer ID
    gPickedIndex = int(data[1])*255 + int(data[0]);
    
    // Uncomment these lines to see the picking shader in effect
    //glfwSwapBuffers(window);
}

void moveVertex(void) {
    //glm::mat4 ModelMatrix = glm::mat4(1.0);
    //GLint viewport[4];
    //glGetIntegerv(GL_VIEWPORT, viewport);
    //glm::vec4 vp = glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]);
    
    // retrieve your cursor position
    // get your world coordinates
    // move points
    
    if(gPickedIndex < CtrlMeshTotalPoints || lastFrameDrag == true){
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        glm::vec3 screenPos = glm::vec3(xpos, 600-ypos, 0.0f);//0.980981
        //glReadPixels(xpos, 600-ypos, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &screenPos.z);
        //glm::vec4 viewport = glm::vec4(0, 0, 600, 600);
        //ModelMatrix = lookAt(cameraPos, cameraLookat, vec3(0.0, 1.0, 0.0));
        //glm::vec3 worldPos = glm::unProject(screenPos, ModelMatrix, gProjectionMatrix, viewport);
        
        static float cursor_x_startPoint = 0.0;
        static float cursor_y_startPoint = 0.0;
        static float cursor_z_startPoint = 0.0;
        static float vertex_x_startPoint = 0.0;
        static float vertex_y_startPoint = 0.0;
        static float vertex_z_startPoint = 0.0;
        
        if(!lastFrameDrag) {
            dragIndex = gPickedIndex;
            cursor_x_startPoint = screenPos[0];
            cursor_y_startPoint = screenPos[1];
            cursor_z_startPoint = screenPos[0];
            
            vertex_x_startPoint = CtrlMeshVerticesVerts[dragIndex].Position[0];
            vertex_y_startPoint = CtrlMeshVerticesVerts[dragIndex].Position[1];
            vertex_z_startPoint = CtrlMeshVerticesVerts[dragIndex].Position[2];
        }
        
        if(!shiftPressed) {      // move vertex on XY plane
            float temp_x = (screenPos[0] - cursor_x_startPoint) / 30.0;
            CtrlMeshVerticesVerts[dragIndex].Position[0] = vertex_x_startPoint + temp_x;
            float temp_y = (screenPos[1] - cursor_y_startPoint) / 30.0;
            CtrlMeshVerticesVerts[dragIndex].Position[1] = vertex_y_startPoint + temp_y;
        }
        else {                  // move vertex in Z-direction
            float temp = (screenPos[0] - cursor_z_startPoint) / 30.0;
            CtrlMeshVerticesVerts[dragIndex].Position[2] = vertex_z_startPoint - temp;
        }
        
        lastFrameDrag = true;
    }
}

int initWindow(void) {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }
    
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    // Open a window and create its OpenGL context
    window = glfwCreateWindow(window_width, window_height, "Ho, Chia-Hsien(90679958)", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
    
    // Set up inputs
    glfwSetCursorPos(window, window_width / 2, window_height / 2);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scroll_callback);
    
    return 0;
}

void initOpenGL(void) {
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);
    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);
    
    // Projection matrix : 45∞ Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    gProjectionMatrix = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
    // Or, for an ortho camera :
    //gProjectionMatrix = glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.0f, 100.0f); // In world coordinates
    
    gViewMatrix = lookAt(cameraPos, cameraLookat, vec3(0.0, 1.0, 0.0));
    
    // Create and compile our GLSL program from the shaders
    programID = LoadShaders("StandardShading.vertexshader", "StandardShading.fragmentshader");
    pickingProgramID = LoadShaders("Picking.vertexshader", "Picking.fragmentshader");
    textrueProgramID = LoadShaders("TextureShading.vertexshader", "TextureShading.fragmentshader");
    
    ModelMatrixID = glGetUniformLocation(programID, "M");
    ViewMatrixID = glGetUniformLocation(programID, "V");
    ProjMatrixID = glGetUniformLocation(programID, "P");
    
    PickingModelMatrixID = glGetUniformLocation(pickingProgramID, "M");
    PickingViewMatrixID = glGetUniformLocation(pickingProgramID, "V");
    PickingProjMatrixID = glGetUniformLocation(pickingProgramID, "P");
    ObjectColorID = glGetUniformLocation(pickingProgramID, "objColor");
    
    TextureModelMatrixID = glGetUniformLocation(textrueProgramID, "M");
    TextureViewMatrixID = glGetUniformLocation(textrueProgramID, "V");
    TextureProjMatrixID = glGetUniformLocation(textrueProgramID, "P");
    TextureID = glGetUniformLocation(textrueProgramID, "tex");
    
    LightID = glGetUniformLocation(programID, "LightPosition_worldspace");
    
    createObjects();
}

void createVAOs(Vertex Vertices[], unsigned short Indices[], int ObjectId) {
    GLenum ErrorCheckValue = glGetError();
    const size_t VertexSize = sizeof(Vertices[0]);
    const size_t RgbOffset = sizeof(Vertices[0].Position);
    const size_t Normaloffset = sizeof(Vertices[0].Color) + RgbOffset;
    const size_t TexCoordoffset = sizeof(Vertices[0].Normal) + Normaloffset;
    
    // Create Vertex Array Object
    glGenVertexArrays(1, &VertexArrayId[ObjectId]);	//
    glBindVertexArray(VertexArrayId[ObjectId]);		//
    
    // Create Buffer for vertex data
    glGenBuffers(1, &VertexBufferId[ObjectId]);
    glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[ObjectId]);
    glBufferData(GL_ARRAY_BUFFER, VertexBufferSize[ObjectId], Vertices, GL_STATIC_DRAW);
    
    // Create Buffer for indices
    if (Indices != NULL) {
        glGenBuffers(1, &IndexBufferId[ObjectId]);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ObjectId]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, IndexBufferSize[ObjectId], Indices, GL_STATIC_DRAW);
    }
    
    // Assign vertex attributes
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)Normaloffset);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)TexCoordoffset);
    
    glEnableVertexAttribArray(0);	// position
    glEnableVertexAttribArray(1);	// color
    glEnableVertexAttribArray(2);	// normal
    glEnableVertexAttribArray(3);	// texture coordinate
    
    // Disable our Vertex Buffer Object
    glBindVertexArray(0);
    
    ErrorCheckValue = glGetError();
    if (ErrorCheckValue != GL_NO_ERROR)
    {
        fprintf(
                stderr,
                "ERROR: Could not create a VBO: %s \n",
                gluErrorString(ErrorCheckValue)
                );
    }
}

void cleanup(void) {
    // Cleanup VBO and shader
    for (int i = 0; i < NumObjects; i++) {
        glDeleteBuffers(1, &VertexBufferId[i]);
        glDeleteBuffers(1, &IndexBufferId[i]);
        glDeleteVertexArrays(1, &VertexArrayId[i]);
    }
    glDeleteProgram(programID);
    glDeleteProgram(pickingProgramID);
    
    delete FaceVerts;
    delete FaceIdcs;
    
    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // ATTN: MODIFY AS APPROPRIATE
    
    if(action == GLFW_RELEASE) {
        if(key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
        shiftPressed = false;
    }
    
    if(action == GLFW_PRESS || action == GLFW_REPEAT) {
        if(key == GLFW_KEY_UP || key == GLFW_KEY_DOWN || key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)
        cameraRotation(key, 0.0);
    }
    
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_C:
            showControlMesh = (showControlMesh)? false : true;
            break;
            case GLFW_KEY_F:
            showFace = (showFace)? false : true;
            break;
            case GLFW_KEY_T:
            showTexture = (showTexture)? false : true;
            break;
            case GLFW_KEY_R:
            cameraRotation(key, 0.0);
            create_original_CtrlMesh();
            break;
            case GLFW_KEY_A:
            if(shiftPressed)
            automateFitting();
            else
            automateFitting_z();
            break;
            case GLFW_KEY_B:
            subdivision();
            break;
            case GLFW_KEY_S:
            exportPoints();
            break;
            case GLFW_KEY_L:
            importPoints();
            break;
            case GLFW_KEY_1:
            switchProfile(GLFW_KEY_1);
            break;
            case GLFW_KEY_2:
            switchProfile(GLFW_KEY_2);
            break;
            case GLFW_KEY_COMMA:
            animate_smile = true;
            break;
            case GLFW_KEY_PERIOD:
            animate_upset = true;
            break;
            case GLFW_KEY_SLASH:
            animate_frown = true;
            break;
            case GLFW_KEY_LEFT_SHIFT:
            shiftPressed = true;
            break;
            case GLFW_KEY_RIGHT_SHIFT:
            shiftPressed = true;
            break;
            
            default:
            break;
        }
    }
}

static void mouseCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
    pickObject();
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cameraRotation(3, yoffset);
}

void cameraRotation(int key, double scroll_offset) {
    static glm::mat4 CamLatitudeRotate = glm::mat4(1.0);
    static glm::mat4 CamLongitudeRotate = glm::mat4(1.0);
    static float CamDistance = 21.2;
    
    if(key == GLFW_KEY_R) {
        CamLatitudeRotate = glm::mat4(1.0);
        CamLongitudeRotate = glm::mat4(1.0);
        CamDistance = 21.2;
        cameraRedDegree = 36;
    }
    else {
        switch(key) {
            case GLFW_KEY_UP:
            CamLatitudeRotate = toMat4(quat(angleAxis(float(-camera_rotate_speed * M_PI / 180.0), normalize(vec3(1.0, 0.0, -1.0))))) * CamLatitudeRotate;
            cameraRedDegree = (cameraRedDegree + int(camera_rotate_speed) + 360) % 360;
            break;
            case GLFW_KEY_DOWN:
            CamLatitudeRotate = toMat4(quat(angleAxis(float(camera_rotate_speed * M_PI / 180.0), normalize(vec3(1.0, 0.0, -1.0))))) * CamLatitudeRotate;
            cameraRedDegree = (cameraRedDegree - int(camera_rotate_speed) + 360) % 360;
            break;
            case GLFW_KEY_LEFT:
            CamLongitudeRotate = toMat4(quat(angleAxis(float(-camera_rotate_speed * M_PI / 180.0), vec3(0.0, 1.0, 0.0)))) * CamLongitudeRotate;
            break;
            case GLFW_KEY_RIGHT:
            CamLongitudeRotate = toMat4(quat(angleAxis(float(camera_rotate_speed * M_PI / 180.0), vec3(0.0, 1.0, 0.0)))) * CamLongitudeRotate;
            break;
            case 3:
            CamDistance += float(scroll_offset) * 0.3f;
            break;
            
            default:
            break;
        }
    }
    
    CamRotationMatrix = CamLongitudeRotate * CamLatitudeRotate;
    
    vec3 dir = normalize(-vec3(CamRotationMatrix * vec4(cameraPos, 1.0)));
    CamTranslationMatrix = translate(glm::mat4(1.0), dir * (CamDistance-21.2f));
    
    vec3 currentCamPos = vec3(CamTranslationMatrix * CamRotationMatrix * vec4(cameraPos, 1.0));
    vec3 tempPosWithoutLatitudeRotate = vec3(CamLongitudeRotate * vec4(cameraPos, 1.0));
    
    if((cameraRedDegree >= 0 && cameraRedDegree < 90) || (cameraRedDegree > 270 && cameraRedDegree < 360))
    gViewMatrix = lookAt(currentCamPos, cameraLookat, vec3(0.0, 1.0, 0.0));
    else if(cameraRedDegree == 90)
    gViewMatrix = lookAt(currentCamPos, cameraLookat, vec3(-tempPosWithoutLatitudeRotate[0], 0.0, -tempPosWithoutLatitudeRotate[2]));
    
    else if(cameraRedDegree == 270)
    gViewMatrix = lookAt(currentCamPos, cameraLookat, vec3(tempPosWithoutLatitudeRotate[0], 0.0, tempPosWithoutLatitudeRotate[2]));
    else
    gViewMatrix = lookAt(currentCamPos, cameraLookat, vec3(0.0, -1.0, 0.0));
}

void automateFitting_z() {
    if(showSubdivision) {
        showSubdivision = false;
        create_original_CtrlMesh();
    }
    
    float scanPlane_z = -3.0;
    float ray_dir[3] = {0.0, 0.0, -1.0};
    
    for(int i = 0; i < CtrlMeshTotalPoints; i++) {
        float pos[3] = {0.0, 0.0, -100.0};
        float normal[3];
        
        for(int j = 0; j < IndexBufferSize[Face]/sizeof(GLushort); j += 3) {
            bool counter_clockwise[3];
            counter_clockwise[0] = CCW_test(FaceVerts[FaceIdcs[j]].Position, FaceVerts[FaceIdcs[j+1]].Position, CtrlMeshVerticesVerts[i].Position);
            counter_clockwise[1] = CCW_test(FaceVerts[FaceIdcs[j+1]].Position, FaceVerts[FaceIdcs[j+2]].Position, CtrlMeshVerticesVerts[i].Position);
            counter_clockwise[2] = CCW_test(FaceVerts[FaceIdcs[j+2]].Position, FaceVerts[FaceIdcs[j]].Position, CtrlMeshVerticesVerts[i].Position);
            
            if((counter_clockwise[0] && counter_clockwise[1] && counter_clockwise[2]) || (!counter_clockwise[0] && !counter_clockwise[1] && !counter_clockwise[2])) {
                float barycentic[3];
                float temp[3] = {0.0, 0.0, 0.0};
                ray_cast(FaceVerts[FaceIdcs[j]].Position, FaceVerts[FaceIdcs[j+1]].Position, FaceVerts[FaceIdcs[j+2]].Position, CtrlMeshVerticesVerts[i].Position, ray_dir, barycentic);
                
                temp[0] = FaceVerts[FaceIdcs[j]].Position[0]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[0]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[0]*barycentic[2];
                temp[1] = FaceVerts[FaceIdcs[j]].Position[1]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[1]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[1]*barycentic[2];
                temp[2] = FaceVerts[FaceIdcs[j]].Position[2]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[2]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[2]*barycentic[2];
                
                if(temp[2] > pos[2]) {
                    pos[0] = temp[0];
                    pos[1] = temp[1];
                    pos[2] = temp[2];
                    
                    normal[0] = (FaceVerts[FaceIdcs[j]].Normal[0] + FaceVerts[FaceIdcs[j+1]].Normal[0] + FaceVerts[FaceIdcs[j+2]].Normal[0]) / 3;
                    normal[1] = (FaceVerts[FaceIdcs[j]].Normal[1] + FaceVerts[FaceIdcs[j+1]].Normal[1] + FaceVerts[FaceIdcs[j+2]].Normal[1]) / 3;
                    normal[2] = (FaceVerts[FaceIdcs[j]].Normal[2] + FaceVerts[FaceIdcs[j+1]].Normal[2] + FaceVerts[FaceIdcs[j+2]].Normal[2]) / 3;
                }
            }
        }
        
        if(pos[2] > scanPlane_z) {
            CtrlMeshVerticesVerts[i].SetPosition(pos);
            CtrlMeshVerticesVerts[i].SetNormal(normal);
        }
        else
        CtrlMeshVerticesVerts[i].Position[2] = scanPlane_z;
    }
}

void automateFitting() {
    if(showSubdivision) {
        showSubdivision = false;
        create_original_CtrlMesh();
    }
    
    float scanPlane_z = -4.0;
    float ray_dir[3];
    float vanishing_line[3] = {0.0, 0.0, -30.0};    // regardless of y
    float barycentic[3];
    
    for(int i = 0; i < CtrlMeshTotalPoints; i++) {
        float pos[3] = {0.0, 0.0, -100.0};
        float normal[3];
        ray_dir[1] = 0.0;
        ray_dir[0] = vanishing_line[0] - CtrlMeshVerticesVerts[i].Position[0];
        ray_dir[2] = vanishing_line[2] - CtrlMeshVerticesVerts[i].Position[2];
        
        for(int j = 0; j < IndexBufferSize[Face]/sizeof(GLushort); j += 3) {
            ray_cast(FaceVerts[FaceIdcs[j]].Position, FaceVerts[FaceIdcs[j+1]].Position, FaceVerts[FaceIdcs[j+2]].Position, CtrlMeshVerticesVerts[i].Position, ray_dir, barycentic);
            
            if((barycentic[0] >= 0.0 && barycentic[0] <= 1.0) && (barycentic[1] >= 0.0 && barycentic[1] <= 1.0) && (barycentic[2] >= 0.0 && barycentic[2] <= 1.0)) {
                float temp[3] = {0.0, 0.0, 0.0};
                
                temp[2] = FaceVerts[FaceIdcs[j]].Position[2]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[2]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[2]*barycentic[2];
                if(temp[2] > pos[2]) {
                    temp[0] = FaceVerts[FaceIdcs[j]].Position[0]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[0]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[0]*barycentic[2];
                    temp[1] = FaceVerts[FaceIdcs[j]].Position[1]*barycentic[0] + FaceVerts[FaceIdcs[j+1]].Position[1]*barycentic[1] + FaceVerts[FaceIdcs[j+2]].Position[1]*barycentic[2];
                    
                    pos[0] = temp[0];
                    pos[1] = temp[1];
                    pos[2] = temp[2];
                    
                    normal[0] = (FaceVerts[FaceIdcs[j]].Normal[0] + FaceVerts[FaceIdcs[j+1]].Normal[0] + FaceVerts[FaceIdcs[j+2]].Normal[0]) / 3;
                    normal[1] = (FaceVerts[FaceIdcs[j]].Normal[1] + FaceVerts[FaceIdcs[j+1]].Normal[1] + FaceVerts[FaceIdcs[j+2]].Normal[1]) / 3;
                    normal[2] = (FaceVerts[FaceIdcs[j]].Normal[2] + FaceVerts[FaceIdcs[j+1]].Normal[2] + FaceVerts[FaceIdcs[j+2]].Normal[2]) / 3;
                }
            }
        }
        
        if(pos[2] > scanPlane_z) {
            CtrlMeshVerticesVerts[i].SetPosition(pos);
            CtrlMeshVerticesVerts[i].SetNormal(normal);
        }
        else
        CtrlMeshVerticesVerts[i].Position[2] = scanPlane_z;
    }
}

bool CCW_test(float *p, float *q, float *r) {
    if((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]) > 0)
    return true;
    else
    return false;
}

void create_original_CtrlMesh() {
    showSubdivision = false;
    PointSize = 5.0;
    
    if(CtrlMeshTotalPoints != 0) {
        delete CtrlMeshVerticesVerts;
        delete CtrlMeshVerticesIdcs_horizontal;
        delete CtrlMeshVerticesIdcs_vertical;
        delete CtrlMeshVerticesIdcs_texture;
        
        glDeleteBuffers(1, &VertexBufferId[CtrlMesh]);
        glDeleteBuffers(1, &IndexBufferId[CtrlMesh]);
        glDeleteVertexArrays(1, &VertexArrayId[CtrlMesh]);
    }
    
    // initialize
    CtrlMeshTotalPoints = pow(initialMeshSide, 2);
    CtrlMeshSide = initialMeshSide;
    CtrlMeshVerticesVerts = new Vertex[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_horizontal = new GLushort[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_vertical = new GLushort[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_texture = new GLushort[int(pow(initialMeshSide-1, 2)*2*3)];
    
    // set world coordinates and texture coordinates
    for(int y = initialMeshSide-1; y >= 0; y--) {
        for(int x = 0; x < initialMeshSide; x++) {
            float coordinate[4] = {((float)x * 20.0/(initialMeshSide-1) - 10.0) * 1.75 + ctrlMesh_x_offset, ((float)y * 20.0/(initialMeshSide-1)) + ctrlMesh_y_offset, 10.0, 1.0};
            CtrlMeshVerticesVerts[(initialMeshSide-1-y)*initialMeshSide + x].SetPosition(coordinate);
            float texCoordinate[2] = {(float(x)/(initialMeshSide-1))*float(texture_s_endPoint-texture_s_startPoint) + texture_s_startPoint, (float(y)/(initialMeshSide-1))*float(texture_t_endPoint-texture_t_startPoint) + texture_t_startPoint};
            CtrlMeshVerticesVerts[(initialMeshSide-1-y)*initialMeshSide + x].SetTexCoord(texCoordinate);
        }
    }
    
    // set color and normal
    float CtrlMeshVerticesColor[4] = {0.0, 1.0, 0.0, 1.0};
    float CtrlMeshVerticesNormal[3] = {0.0, 0.0, -1.0};
    for(int i = 0; i < CtrlMeshTotalPoints; i++) {
        CtrlMeshVerticesVerts[i].SetColor(CtrlMeshVerticesColor);
        CtrlMeshVerticesVerts[i].SetNormal(CtrlMeshVerticesNormal);
    }
    
    // set three indices arrays
    for(int i = 0; i < CtrlMeshTotalPoints; i++)
    CtrlMeshVerticesIdcs_vertical[i] = i;
    for(int i = 0; i < initialMeshSide; i++) {
        for(int j = 0; j < initialMeshSide; j++)
        CtrlMeshVerticesIdcs_horizontal[i*initialMeshSide + j] = j*initialMeshSide + i;
    }
    
    for(int i = 0; i < initialMeshSide-1; i++) {
        for(int j = 0; j < initialMeshSide-1; j++) {
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6] = i*initialMeshSide + j;
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6 + 1] = i*initialMeshSide + j + 1;
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6 + 2] = (i+1)*initialMeshSide + j;
            
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6 + 3] = i*initialMeshSide + j + 1;
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6 + 4] = (i+1)*initialMeshSide + j;
            CtrlMeshVerticesIdcs_texture[i*6*(initialMeshSide-1) + j*6 + 5] = (i+1)*initialMeshSide + j + 1;
        }
    }
    
    // create VAO
    VertexBufferSize[CtrlMesh] = sizeof(Vertex) * CtrlMeshTotalPoints;
    NumIndices[CtrlMesh] = int(pow(initialMeshSide-1, 2)*2*3);
    IndexBufferSize[CtrlMesh] = sizeof(GLushort) * NumIndices[CtrlMesh];
    createVAOs(CtrlMeshVerticesVerts, CtrlMeshVerticesIdcs_vertical, CtrlMesh);
}

void subdivision() {
    showSubdivision = true;
    PointSize = 3.0;
    
    Vertex *old_CtrlMeshVerticesVerts = CtrlMeshVerticesVerts;
    
    delete CtrlMeshVerticesIdcs_horizontal;
    delete CtrlMeshVerticesIdcs_vertical;
    delete CtrlMeshVerticesIdcs_texture;
    
    glDeleteBuffers(1, &VertexBufferId[CtrlMesh]);
    glDeleteBuffers(1, &IndexBufferId[CtrlMesh]);
    glDeleteVertexArrays(1, &VertexArrayId[CtrlMesh]);
    
    // initialize
    CtrlMeshSide *= 3;
    CtrlMeshTotalPoints = pow(CtrlMeshSide, 2);
    const int triangulateIndexCount = int(pow(CtrlMeshSide, 2)) * 2 * 3;
    CtrlMeshVerticesVerts = new Vertex[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_horizontal = new GLushort[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_vertical = new GLushort[CtrlMeshTotalPoints];
    CtrlMeshVerticesIdcs_texture = new GLushort[triangulateIndexCount];
    
    // set color, world coordinates, and normal
    int old_side = CtrlMeshSide / 3;
    float red[4] = {1.0, 0.0, 0.0, 1.0};
    float green[4] = {0.0, 1.0, 0.0, 1.0};
    float blue[4] = {0.0, 0.0, 1.0, 1.0};
    float black[4] = {0.0, 0.0, 0.0, 1.0};
    
    for(int i = 0; i < pow(old_side, 2); i++) {
        if(!((i%old_side == 0) || ((i+1)%old_side == 0) || (i < old_side) || (i >= old_side*(old_side-1)))) {  // not margin
            // upper left
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetColor(blue);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetPosition(
                                                                                            ((point(old_CtrlMeshVerticesVerts[center].Position)*4 +
                                                                                              (point(old_CtrlMeshVerticesVerts[left].Position) +
                                                                                               point(old_CtrlMeshVerticesVerts[up].Position))*2 +
                                                                                              point(old_CtrlMeshVerticesVerts[up_left].Position)) / 9).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetNormal(
                                                                                          ((point(old_CtrlMeshVerticesVerts[center].Normal)*4 +
                                                                                            (point(old_CtrlMeshVerticesVerts[left].Normal) +
                                                                                             point(old_CtrlMeshVerticesVerts[up].Normal))*2 +
                                                                                            point(old_CtrlMeshVerticesVerts[up_left].Normal)) / 9).toArray());
            // up
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetColor(green);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetPosition(
                                                                                                ((point(old_CtrlMeshVerticesVerts[center].Position)*8 +
                                                                                                  (point(old_CtrlMeshVerticesVerts[left].Position) +
                                                                                                   point(old_CtrlMeshVerticesVerts[right].Position))*2 +
                                                                                                  point(old_CtrlMeshVerticesVerts[up].Position)*4 +
                                                                                                  point(old_CtrlMeshVerticesVerts[up_left].Position) +
                                                                                                  point(old_CtrlMeshVerticesVerts[up_right].Position)) / 18).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetNormal(
                                                                                              ((point(old_CtrlMeshVerticesVerts[center].Normal)*8 +
                                                                                                (point(old_CtrlMeshVerticesVerts[left].Normal) +
                                                                                                 point(old_CtrlMeshVerticesVerts[right].Normal))*2 +
                                                                                                point(old_CtrlMeshVerticesVerts[up].Normal)*4 +
                                                                                                point(old_CtrlMeshVerticesVerts[up_left].Normal) +
                                                                                                point(old_CtrlMeshVerticesVerts[up_right].Normal)) / 18).toArray());
            // upper right
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetColor(blue);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetPosition(
                                                                                                ((point(old_CtrlMeshVerticesVerts[center].Position)*4 +
                                                                                                  (point(old_CtrlMeshVerticesVerts[up].Position) +
                                                                                                   point(old_CtrlMeshVerticesVerts[right].Position))*2 +
                                                                                                  point(old_CtrlMeshVerticesVerts[up_right].Position)) / 9).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetNormal(
                                                                                              ((point(old_CtrlMeshVerticesVerts[center].Normal)*4 +
                                                                                                (point(old_CtrlMeshVerticesVerts[up].Normal) +
                                                                                                 point(old_CtrlMeshVerticesVerts[right].Normal))*2 +
                                                                                                point(old_CtrlMeshVerticesVerts[up_right].Normal)) / 9).toArray());
            // left
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetColor(green);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetPosition(
                                                                                                           ((point(old_CtrlMeshVerticesVerts[center].Position)*8 +
                                                                                                             (point(old_CtrlMeshVerticesVerts[up].Position) +
                                                                                                              point(old_CtrlMeshVerticesVerts[down].Position))*2 +
                                                                                                             point(old_CtrlMeshVerticesVerts[left].Position)*4 +
                                                                                                             point(old_CtrlMeshVerticesVerts[down_left].Position) +
                                                                                                             point(old_CtrlMeshVerticesVerts[up_left].Position)) / 18).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetNormal(
                                                                                                         ((point(old_CtrlMeshVerticesVerts[center].Normal)*8 +
                                                                                                           (point(old_CtrlMeshVerticesVerts[up].Normal) +
                                                                                                            point(old_CtrlMeshVerticesVerts[down].Normal))*2 +
                                                                                                           point(old_CtrlMeshVerticesVerts[left].Normal)*4 +
                                                                                                           point(old_CtrlMeshVerticesVerts[down_left].Normal) +
                                                                                                           point(old_CtrlMeshVerticesVerts[up_left].Normal)) / 18).toArray());
            // center
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetColor(red);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetPosition(
                                                                                                               ((point(old_CtrlMeshVerticesVerts[center].Position)*16 +
                                                                                                                 (point(old_CtrlMeshVerticesVerts[up].Position) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[down].Position) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[left].Position) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[right].Position))*4 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[up_left].Position) +
                                                                                                                 point(old_CtrlMeshVerticesVerts[up_right].Position) +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_left].Position) +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_right].Position)) / 36).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetNormal(
                                                                                                             ((point(old_CtrlMeshVerticesVerts[center].Normal)*16 +
                                                                                                               (point(old_CtrlMeshVerticesVerts[up].Normal) +
                                                                                                                point(old_CtrlMeshVerticesVerts[down].Normal) +
                                                                                                                point(old_CtrlMeshVerticesVerts[left].Normal) +
                                                                                                                point(old_CtrlMeshVerticesVerts[right].Normal))*4 +
                                                                                                               point(old_CtrlMeshVerticesVerts[up_left].Normal) +
                                                                                                               point(old_CtrlMeshVerticesVerts[up_right].Normal) +
                                                                                                               point(old_CtrlMeshVerticesVerts[down_left].Normal) +
                                                                                                               point(old_CtrlMeshVerticesVerts[down_right].Normal)) / 36).toArray());
            // right
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetColor(green);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetPosition(
                                                                                                               ((point(old_CtrlMeshVerticesVerts[center].Position)*8 +
                                                                                                                 (point(old_CtrlMeshVerticesVerts[up].Position) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[down].Position))*2 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[right].Position)*4 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[up_right].Position) +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_right].Position)) / 18).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetNormal(
                                                                                                             ((point(old_CtrlMeshVerticesVerts[center].Normal)*8 +
                                                                                                               (point(old_CtrlMeshVerticesVerts[up].Normal) +
                                                                                                                point(old_CtrlMeshVerticesVerts[down].Normal))*2 +
                                                                                                               point(old_CtrlMeshVerticesVerts[right].Normal)*4 +
                                                                                                               point(old_CtrlMeshVerticesVerts[up_right].Normal) +
                                                                                                               point(old_CtrlMeshVerticesVerts[down_right].Normal)) / 18).toArray());
            // lower left
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetColor(blue);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetPosition(
                                                                                                             ((point(old_CtrlMeshVerticesVerts[center].Position)*4 +
                                                                                                               (point(old_CtrlMeshVerticesVerts[down].Position) +
                                                                                                                point(old_CtrlMeshVerticesVerts[left].Position))*2 +
                                                                                                               point(old_CtrlMeshVerticesVerts[down_left].Position)) / 9).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetNormal(
                                                                                                           ((point(old_CtrlMeshVerticesVerts[center].Normal)*4 +
                                                                                                             (point(old_CtrlMeshVerticesVerts[down].Normal) +
                                                                                                              point(old_CtrlMeshVerticesVerts[left].Normal))*2 +
                                                                                                             point(old_CtrlMeshVerticesVerts[down_left].Normal)) / 9).toArray());
            // down
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetColor(green);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetPosition(
                                                                                                                 ((point(old_CtrlMeshVerticesVerts[center].Position)*8 +
                                                                                                                   (point(old_CtrlMeshVerticesVerts[left].Position) +
                                                                                                                    point(old_CtrlMeshVerticesVerts[right].Position))*2 +
                                                                                                                   point(old_CtrlMeshVerticesVerts[down].Position)*4 +
                                                                                                                   point(old_CtrlMeshVerticesVerts[down_left].Position) +
                                                                                                                   point(old_CtrlMeshVerticesVerts[down_right].Position)) / 18).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetNormal(
                                                                                                               ((point(old_CtrlMeshVerticesVerts[center].Normal)*8 +
                                                                                                                 (point(old_CtrlMeshVerticesVerts[left].Normal) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[right].Normal))*2 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down].Normal)*4 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_left].Normal) +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_right].Normal)) / 18).toArray());
            // lower right
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetColor(blue);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetPosition(
                                                                                                                 ((point(old_CtrlMeshVerticesVerts[center].Position)*4 +
                                                                                                                   (point(old_CtrlMeshVerticesVerts[right].Position) +
                                                                                                                    point(old_CtrlMeshVerticesVerts[down].Position))*2 +
                                                                                                                   point(old_CtrlMeshVerticesVerts[down_right].Position)) / 9).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetNormal(
                                                                                                               ((point(old_CtrlMeshVerticesVerts[center].Normal)*4 +
                                                                                                                 (point(old_CtrlMeshVerticesVerts[right].Normal) +
                                                                                                                  point(old_CtrlMeshVerticesVerts[down].Normal))*2 +
                                                                                                                 point(old_CtrlMeshVerticesVerts[down_right].Normal)) / 9).toArray());
        }
        else {
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetColor(black);
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetColor(black);
            
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetPosition(point(old_CtrlMeshVerticesVerts[i].Position).toArray());
            
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3)].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 1].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + 2].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 1].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide + 2].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 1].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
            CtrlMeshVerticesVerts[((i/old_side)*9*old_side) + ((i%old_side)*3) + CtrlMeshSide*2 + 2].SetNormal(point(old_CtrlMeshVerticesVerts[i].Normal).toArray());
        }
    }
    
    delete old_CtrlMeshVerticesVerts;
    
    // set texture coordinates
    for(int y = CtrlMeshSide-1; y >= 0; y--) {
        for(int x = 0; x < CtrlMeshSide; x++) {
            float texCoordinate[2] = {(float(x)/(CtrlMeshSide-1))*float(texture_s_endPoint-texture_s_startPoint) + texture_s_startPoint, (float(y)/(CtrlMeshSide-1))*float(texture_t_endPoint-texture_t_startPoint) + texture_t_startPoint};
            CtrlMeshVerticesVerts[(CtrlMeshSide-1-y)*CtrlMeshSide + x].SetTexCoord(texCoordinate);
        }
    }
    
    // set three indices arrays
    for(int i = 0; i < CtrlMeshTotalPoints; i++)
    CtrlMeshVerticesIdcs_vertical[i] = i;
    for(int i = 0; i < CtrlMeshSide; i++) {
        for(int j = 0; j < CtrlMeshSide; j++)
        CtrlMeshVerticesIdcs_horizontal[i * (CtrlMeshSide) + j] = j * (CtrlMeshSide) + i;
    }
    
    for(int i = 0; i < CtrlMeshSide-1; i++) {
        for(int j = 0; j < CtrlMeshSide-1; j++) {
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6] = i*CtrlMeshSide + j;
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6 + 1] = i*CtrlMeshSide + j + 1;
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6 + 2] = (i+1)*CtrlMeshSide + j;
            
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6 + 3] = i*CtrlMeshSide+ + j + 1;
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6 + 4] = (i+1)*CtrlMeshSide + j;
            CtrlMeshVerticesIdcs_texture[i*(CtrlMeshSide-1)*6 + j*6 + 5] = (i+1)*CtrlMeshSide + j + 1;
        }
    }
    
    // create VAO
    VertexBufferSize[CtrlMesh] = sizeof(Vertex) * CtrlMeshTotalPoints;
    NumIndices[CtrlMesh] = triangulateIndexCount;
    IndexBufferSize[CtrlMesh] = sizeof(GLushort) * triangulateIndexCount;
    createVAOs(CtrlMeshVerticesVerts, CtrlMeshVerticesIdcs_vertical, CtrlMesh);
}

void exportPoints() {
    ofstream out("cm.p3", ofstream::out);
    if(!out) {
        cout << "Error opening file." << endl;
        return;
    }
    
    if(showSubdivision) {
        cout << "Cannot store control points after subdivision." << endl;
        return;
    }
    
    for(int i = 0; i < pow(initialMeshSide, 2); i++) {
        out << setw(10) << CtrlMeshVerticesVerts[i].Position[0] << " " << setw(10) << CtrlMeshVerticesVerts[i].Position[1] << " " << setw(10) << CtrlMeshVerticesVerts[i].Position[2] << " ";
        out << setw(10) << CtrlMeshVerticesVerts[i].Normal[0] << " " << setw(10) << CtrlMeshVerticesVerts[i].Normal[1] << " " << setw(10) << CtrlMeshVerticesVerts[i].Normal[2] << endl;
    }
    
    out.close();
}

void importPoints() {
    create_original_CtrlMesh();
    
    ifstream in("cm.p3", ifstream::in);
    if(!in) {
        cout << setw << "Error opening file." << endl;
        return;
    }
    
    string line;
    for(int i = 0; i < pow(initialMeshSide, 2); i++) {
        getline(in, line);
        
        string temp;
        temp = line.substr(0, 10);
        CtrlMeshVerticesVerts[i].Position[0] = stof(temp);
        temp = line.substr(11, 10);
        CtrlMeshVerticesVerts[i].Position[1] = stof(temp);
        temp = line.substr(22, 10);
        CtrlMeshVerticesVerts[i].Position[2] = stof(temp);
        
        temp = line.substr(33, 10);
        CtrlMeshVerticesVerts[i].Normal[0] = stof(temp);
        temp = line.substr(44, 10);
        CtrlMeshVerticesVerts[i].Normal[1] = stof(temp);
        temp = line.substr(55, 10);
        CtrlMeshVerticesVerts[i].Normal[2] = stof(temp);
    }
    
    in.close();
}

void switchProfile(int profile) {
    mat4 faceTranslate = mat4(1.0);
    mat4 faceRotate = mat4(1.0);
    mat4 faceScale = mat4(1.0);
    long height = 457, width = 360;
    
    switch(profile) {
        case GLFW_KEY_1:
        loadObject("models/ho_chiahsien.obj", glm::vec4(1.0, 0.85, 0.75, 1.0), FaceVerts, FaceIdcs, Face);
        TextureID = load_texture_TGA("models/ho_chiahsien.tga", &height, &width, GL_CLAMP, GL_CLAMP);
        
        faceTranslate = translate(faceTranslate, vec3(0.0f, 0.0f, -5.0f));
        faceRotate = faceRotate * toMat4(quat(angleAxis(3.3f, vec3(0.0, 1.0, 0.0))));
        faceScale = scale(faceScale, vec3(70.0f));
        
        ctrlMesh_x_offset = 0.5;
        ctrlMesh_y_offset = 0.0;
        texture_s_startPoint = 0.05;
        texture_s_endPoint = 0.95;
        texture_t_startPoint = 0.0;
        texture_t_endPoint = 0.9;
        left_mouth_corner_x = 0.42;
        left_mouth_corner_y = 0.22;
        right_mouth_corner_x = 0.5;
        right_mouth_corner_y = 0.22;
        left_eyebrow_center_x = 0.42;
        left_eyebrow_center_y = 0.55;
        right_eyebrow_center_x = 0.5;
        right_eyebrow_center_y = 0.55;
        break;
        
        case GLFW_KEY_2:
        loadObject("models/chihyin_lee.obj", glm::vec4(1.0, 0.85, 0.75, 1.0), FaceVerts, FaceIdcs, Face);
        TextureID = load_texture_TGA("models/chihyin_lee.tga", &height, &width, GL_CLAMP, GL_CLAMP);
        
        faceTranslate = translate(faceTranslate, vec3(0.0f, 0.0f, -3.0f));
        faceRotate = faceRotate * toMat4(quat(angleAxis(3.3f, vec3(0.0, 1.0, 0.0))));
        faceScale = scale(faceScale, vec3(65.0f));
        
        ctrlMesh_x_offset = 0.0;
        ctrlMesh_y_offset = 0.0;
        texture_s_startPoint = 0.0;
        texture_s_endPoint = 0.975;
        texture_t_startPoint = 0.0;
        texture_t_endPoint = 0.94;
        left_mouth_corner_x = 0.38;
        left_mouth_corner_y = 0.23;
        right_mouth_corner_x = 0.49;
        right_mouth_corner_y = 0.23;
        left_eyebrow_center_x = 0.38;
        left_eyebrow_center_y = 0.55;
        right_eyebrow_center_x = 0.48;
        right_eyebrow_center_y = 0.55;
        break;
        
        default:
        break;
    }
    
    mat4 faceModelMatrix = faceTranslate * faceRotate * faceScale;
    
    for(int i = 0; i < VertexBufferSize[Face]/sizeof(Vertex); i++) {
        glm::vec4 temp = faceModelMatrix * glm::vec4(FaceVerts[i].Position[0], FaceVerts[i].Position[1], FaceVerts[i].Position[2], FaceVerts[i].Position[3]);
        for(int j = 0; j < 4; j++)
        FaceVerts[i].Position[j] = temp[j];
    }
    
    createVAOs(FaceVerts, FaceIdcs, Face);
    create_original_CtrlMesh();
}

void smileAnimation() {
    static bool last_frame_animate = false;
    static int currentTime;
    const int timelaspe = 100;
    static int *leftMovePointIndices = NULL;
    static int leftMovePointCount = 0;
    static int *rightMovePointIndices = NULL;
    static int rightMovePointCount = 0;
    
    if(!last_frame_animate) {
        currentTime = 0;
        leftMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        leftMovePointCount = 0;
        rightMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        rightMovePointCount = 0;
        int x = 0, y = 0;
        float x_texCoord = 0.0, y_texCoord = 0.0;
        
        for(int i = 0; i < CtrlMeshTotalPoints; i++) {
            x = i % CtrlMeshSide;
            y = i / CtrlMeshSide;
            x_texCoord = float(x) / CtrlMeshSide;
            y_texCoord = 1.0 - float(y) / CtrlMeshSide;
            
            if((x_texCoord > left_mouth_corner_x-0.03) && (x_texCoord < left_mouth_corner_x+0.03) && (y_texCoord > left_mouth_corner_y-0.05) && (y_texCoord < left_mouth_corner_y+0.05))
            leftMovePointIndices[leftMovePointCount++] = i;
            else if((x_texCoord > right_mouth_corner_x-0.03) && (x_texCoord < right_mouth_corner_x+0.03) && (y_texCoord > right_mouth_corner_y-0.05) && (y_texCoord < right_mouth_corner_y+0.05))
            rightMovePointIndices[rightMovePointCount++] = i;
        }
        
        /*showSubdivision = true;
         float color[4] = {1.0, 0.0, 0.0, 1.0};
         for(int i = 0; i < leftMovePointCount; i++)
         CtrlMeshVerticesVerts[leftMovePointIndices[i]].SetColor(color);
         for(int i = 0; i < rightMovePointCount; i++)
         CtrlMeshVerticesVerts[rightMovePointIndices[i]].SetColor(color);*/
        
        last_frame_animate = true;
    }
    
    if(currentTime < timelaspe) {
        glm::mat4 leftFrameTranslate = translate(glm::mat4(1.0), vec3(-0.3/timelaspe, 0.25/timelaspe, -0.15/timelaspe));
        glm::mat4 rightFrameTranslate = translate(glm::mat4(1.0), vec3(0.3/timelaspe, 0.25/timelaspe, -0.15/timelaspe));
        
        for(int i = 0; i < leftMovePointCount; i++) {
            glm::vec4 temp = leftFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[j] = temp[j];
        }
        for(int i = 0; i < rightMovePointCount; i++) {
            glm::vec4 temp = rightFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[j] = temp[j];
        }
        
        currentTime++;
    }
    else {
        last_frame_animate = false;
        animate_smile = false;
    }
}

void upsetAnimation() {
    static bool last_frame_animate = false;
    static int currentTime;
    const int timelaspe = 100;
    static int *leftMovePointIndices = NULL;
    static int leftMovePointCount = 0;
    static int *rightMovePointIndices = NULL;
    static int rightMovePointCount = 0;
    
    if(!last_frame_animate) {
        currentTime = 0;
        leftMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        leftMovePointCount = 0;
        rightMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        rightMovePointCount = 0;
        int x = 0, y = 0;
        float x_texCoord = 0.0, y_texCoord = 0.0;
        
        for(int i = 0; i < CtrlMeshTotalPoints; i++) {
            x = i % CtrlMeshSide;
            y = i / CtrlMeshSide;
            x_texCoord = float(x) / CtrlMeshSide;
            y_texCoord = 1.0 - float(y) / CtrlMeshSide;
            
            if((x_texCoord > left_mouth_corner_x-0.04) && (x_texCoord < left_mouth_corner_x+0.03) && (y_texCoord > left_mouth_corner_y-0.06) && (y_texCoord < left_mouth_corner_y+0.08))
            leftMovePointIndices[leftMovePointCount++] = i;
            else if((x_texCoord > right_mouth_corner_x-0.03) && (x_texCoord < right_mouth_corner_x+0.04) && (y_texCoord > right_mouth_corner_y-0.06) && (y_texCoord < right_mouth_corner_y+0.08))
            rightMovePointIndices[rightMovePointCount++] = i;
        }
        
        /*showSubdivision = true;
         float color[4] = {1.0, 0.0, 0.0, 1.0};
         for(int i = 0; i < leftMovePointCount; i++)
         CtrlMeshVerticesVerts[leftMovePointIndices[i]].SetColor(color);
         for(int i = 0; i < rightMovePointCount; i++)
         CtrlMeshVerticesVerts[rightMovePointIndices[i]].SetColor(color);*/
        
        last_frame_animate = true;
    }
    
    if(currentTime < timelaspe) {
        glm::mat4 leftFrameTranslate = translate(glm::mat4(1.0), vec3(-0.2/timelaspe, -0.3/timelaspe, -0.15/timelaspe));
        glm::mat4 rightFrameTranslate = translate(glm::mat4(1.0), vec3(0.2/timelaspe, -0.3/timelaspe, -0.15/timelaspe));
        
        for(int i = 0; i < leftMovePointCount; i++) {
            glm::vec4 temp = leftFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[j] = temp[j];
        }
        for(int i = 0; i < rightMovePointCount; i++) {
            glm::vec4 temp = rightFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[j] = temp[j];
        }
        
        currentTime++;
    }
    else {
        last_frame_animate = false;
        animate_upset = false;
    }
}

void frownAnimation() {
    static bool last_frame_animate = false;
    static int currentTime;
    const int timelaspe = 100;
    static int *leftMovePointIndices = NULL;
    static int leftMovePointCount = 0;
    static int *rightMovePointIndices = NULL;
    static int rightMovePointCount = 0;
    
    if(!last_frame_animate) {
        currentTime = 0;
        leftMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        leftMovePointCount = 0;
        rightMovePointIndices = new int[int(CtrlMeshTotalPoints * 0.04)];
        rightMovePointCount = 0;
        int x = 0, y = 0;
        float x_texCoord = 0.0, y_texCoord = 0.0;
        
        for(int i = 0; i < CtrlMeshTotalPoints; i++) {
            x = i % CtrlMeshSide;
            y = i / CtrlMeshSide;
            x_texCoord = float(x) / CtrlMeshSide;
            y_texCoord = 1.0 - float(y) / CtrlMeshSide;
            
            if((x_texCoord > left_eyebrow_center_x-0.05) && (x_texCoord < left_eyebrow_center_x+0.05) && (y_texCoord > left_eyebrow_center_y-0.05) && (y_texCoord < left_eyebrow_center_y+0.05))
            leftMovePointIndices[leftMovePointCount++] = i;
            else if((x_texCoord > right_eyebrow_center_x-0.06) && (x_texCoord < right_eyebrow_center_x+0.06) && (y_texCoord > right_eyebrow_center_y-0.05) && (y_texCoord < right_eyebrow_center_y+0.05))
            rightMovePointIndices[rightMovePointCount++] = i;
        }
        
        /*showSubdivision = true;
         float color[4] = {1.0, 0.0, 0.0, 1.0};
         for(int i = 0; i < leftMovePointCount; i++)
         CtrlMeshVerticesVerts[leftMovePointIndices[i]].SetColor(color);
         for(int i = 0; i < rightMovePointCount; i++)
         CtrlMeshVerticesVerts[rightMovePointIndices[i]].SetColor(color);*/
        
        last_frame_animate = true;
    }
    
    if(currentTime < timelaspe) {
        glm::mat4 leftFrameTranslate = translate(glm::mat4(1.0), vec3(0.2/timelaspe, -0.2/timelaspe, 0.0/timelaspe));
        glm::mat4 rightFrameTranslate = translate(glm::mat4(1.0), vec3(-0.2/timelaspe, -0.2/timelaspe, 0.0/timelaspe));
        
        for(int i = 0; i < leftMovePointCount; i++) {
            glm::vec4 temp = leftFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[leftMovePointIndices[i]].Position[j] = temp[j];
        }
        for(int i = 0; i < rightMovePointCount; i++) {
            glm::vec4 temp = rightFrameTranslate * glm::vec4(CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[0], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[1], CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[2], 1.0);
            for(int j = 0; j < 4; j++)
            CtrlMeshVerticesVerts[rightMovePointIndices[i]].Position[j] = temp[j];
        }
        
        currentTime++;
    }
    else {
        last_frame_animate = false;
        animate_frown = false;
    }
}

int main(void) {
    // initialize window
    int errorCode = initWindow();
    if (errorCode != 0)
    return errorCode;
    
    // initialize OpenGL pipeline
    initOpenGL();
    
    do {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT))
        moveVertex();
        else
        lastFrameDrag = false;
        
        renderScene();
    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0);
    
    cleanup();
    
    return 0;
}
