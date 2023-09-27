cmake_minimum_required(VERSION 3.23)
project(01_DrawTriangle
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(SHADER_DIRECTORY "Shaders")

set(CMAKE_EXE_LINKER_FLAGS "-static")

find_package(Vulkan REQUIRED)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../Scripts DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(${PROJECT_NAME} 01_DrawTriangle.cpp)

add_custom_target(BuildShaders
        COMMAND Scripts/BuildShaders.bat
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Build Shaders ..."
        VERBATIM
)

add_dependencies(${PROJECT_NAME} BuildShaders)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
