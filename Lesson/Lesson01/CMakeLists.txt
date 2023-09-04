cmake_minimum_required(VERSION 3.23)
project(DrawTriangle
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_executable(DrawTriangle DrawTriangle.cpp)

target_link_libraries(DrawTriangle PRIVATE Vulkan::Vulkan)
target_link_libraries(DrawTriangle PRIVATE glfw)
