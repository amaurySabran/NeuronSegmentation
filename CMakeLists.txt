CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(project)

set (SOURCE_FILES src/.)


ADD_EXECUTABLE(main src/main.cpp  src/Image3D.cpp src/Point3D.cpp   src/maxflow/graph.cpp)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})
