cmake_minimum_required(VERSION 3.14)
project(cvtracker VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# OpenCV (only core + tracking + imgproc needed – no Eigen, no BLAS)
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc tracking)

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------
add_library(cvtracker SHARED
    src/cvtracker.cpp
)

target_include_directories(cvtracker
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(cvtracker PRIVATE CVTRACKER_EXPORTS)

target_link_libraries(cvtracker
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_tracking
)

set_target_properties(cvtracker PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------------------
# Optional: install rules
# ---------------------------------------------------------------------------
install(TARGETS cvtracker
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/cvtracker_c.h DESTINATION include)
