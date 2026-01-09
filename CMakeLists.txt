cmake_minimum_required(VERSION 3.13)

# == Pico SDK ==
include(pico_sdk_import.cmake)

project(cnn_mnist C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# If you pass -DPICO_BOARD=... on the command line, this will be overridden.
set(PICO_BOARD pico_w CACHE STRING "Pico board")

pico_sdk_init()

# ------------------------------------------------------------------------------
# pico-tflmicro
#   Your error log shows Ninja compiling pico-tflmicro *tests* (very long paths)
#   and failing to create dependency files on Windows.
#   So we explicitly disable pico-tflmicro tests/examples/benchmarks and add the
#   subdirectory as EXCLUDE_FROM_ALL.
# ------------------------------------------------------------------------------

# Common CMake test switch
set(BUILD_TESTING OFF CACHE BOOL "Disable building tests" FORCE)

# These option names are used by pico-tflmicro in many versions; if a given
# variable is unknown it is simply ignored by CMake.
set(PICO_TFLMICRO_BUILD_TESTS OFF CACHE BOOL "Disable pico-tflmicro tests" FORCE)
set(PICO_TFLMICRO_BUILD_EXAMPLES OFF CACHE BOOL "Disable pico-tflmicro examples" FORCE)
set(PICO_TFLMICRO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable pico-tflmicro benchmarks" FORCE)

# Add pico-tflmicro (vendored in ./pico-tflmicro)
add_subdirectory(pico-tflmicro pico-tflmicro-build EXCLUDE_FROM_ALL)

# Detect the correct pico-tflmicro library target name (varies by fork/version)
set(TFLM_TARGET "")
foreach(candidate IN ITEMS pico_tflmicro pico-tflmicro tflmicro pico_tflmicro_lib)
    if(TARGET ${candidate})
        set(TFLM_TARGET ${candidate})
        break()
    endif()
endforeach()

if(TFLM_TARGET STREQUAL "")
    message(FATAL_ERROR
        "Could not find a pico-tflmicro library target after add_subdirectory(). "
        "Expected one of: pico_tflmicro, pico-tflmicro, tflmicro, pico_tflmicro_lib."
    )
endif()

# ------------------------------------------------------------------------------
# Your application
# ------------------------------------------------------------------------------

add_executable(cnn_mnist
    cnn_mnist.c
    tflm_wrapper.cpp
)

pico_set_program_name(cnn_mnist "cnn_mnist")
pico_set_program_version(cnn_mnist "0.1")

# I/O
pico_enable_stdio_uart(cnn_mnist 0)
pico_enable_stdio_usb(cnn_mnist 1)

target_include_directories(cnn_mnist PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(cnn_mnist PRIVATE
    pico_stdlib
    pico_cyw43_arch_none
    ${TFLM_TARGET}
)

pico_add_extra_outputs(cnn_mnist)
