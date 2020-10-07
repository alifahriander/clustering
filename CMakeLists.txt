# only for cmake --version >= 3.5.1
cmake_minimum_required(VERSION 3.5.1)

# project name
project(project-clustering)


# set the C++14 standard
set(CMAKE_CXX_STANDARD 14)

# set EXEC
set(EXEC clustering)

# I../includes
include_directories(includes)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# puts all .cpp files inside src to the SOURCES variable
file(GLOB SOURCES src/*.cpp)

# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_executable(${EXEC} ${SOURCES})
target_link_libraries(${EXEC} Eigen3::Eigen)
