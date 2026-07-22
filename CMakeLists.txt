cmake_minimum_required(VERSION 3.20)
project(ncnn_ocr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP REQUIRED)

if(DEFINED ENV{NCNN_DIR} AND NOT ncnn_DIR)
    set(ncnn_DIR "$ENV{NCNN_DIR}/lib/cmake/ncnn")
endif()
if(NOT ncnn_DIR)
    foreach(_hint
        "$ENV{NCNN_DIR}/lib/cmake/ncnn"
        "$ENV{HOME}/.local/ncnn/lib/cmake/ncnn"
        "/opt/homebrew/opt/ncnn/lib/cmake/ncnn"
        "/usr/local/opt/ncnn/lib/cmake/ncnn"
        "/usr/local/lib/cmake/ncnn")
        if(EXISTS "${_hint}/ncnnConfig.cmake")
            set(ncnn_DIR "${_hint}")
            break()
        endif()
    endforeach()
endif()
find_package(ncnn REQUIRED)

set(PADDLEOCR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/native/paddleocr")

set(NCNN_OCR_SOURCES
    native/bindings.cpp
    ${PADDLEOCR_DIR}/db_net.cpp
    ${PADDLEOCR_DIR}/angle_net.cpp
    ${PADDLEOCR_DIR}/crnn_net.cpp
    ${PADDLEOCR_DIR}/ocr_engine.cpp
    ${PADDLEOCR_DIR}/utils.cpp
    ${PADDLEOCR_DIR}/3rdparty/clipper2/clipper.engine.cpp
    ${PADDLEOCR_DIR}/3rdparty/clipper2/clipper.offset.cpp
    ${PADDLEOCR_DIR}/3rdparty/clipper2/clipper.rectclip.cpp
)

pybind11_add_module(_ncnn_ocr MODULE ${NCNN_OCR_SOURCES})

target_include_directories(_ncnn_ocr PRIVATE
    ${PADDLEOCR_DIR}
    ${PADDLEOCR_DIR}/3rdparty
)

target_link_libraries(_ncnn_ocr PRIVATE
    ${OpenCV_LIBS}
    ncnn
    OpenMP::OpenMP_CXX
)

if(WIN32)
    target_compile_definitions(_ncnn_ocr PRIVATE NOMINMAX)
endif()

install(TARGETS _ncnn_ocr DESTINATION ncnn_ocr)
