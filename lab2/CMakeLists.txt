cmake_minimum_required(VERSION 3.10)

project(lab2)

find_package(OpenMP)

add_definitions()
add_compile_options(-fopenmp)
add_executable(lab2 delta-stepping.cc)