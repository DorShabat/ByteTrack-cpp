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
# Bundling: on Windows and Linux copy OpenCV libraries next to the tracker
# so the output folder is self-contained without a system-wide OpenCV install.
# On Apple the system Homebrew install is used as-is (no bundling).
# ---------------------------------------------------------------------------
if(NOT APPLE)
    # Set RPATH so the loader finds the bundled libs in the same directory.
    set_target_properties(cvtracker PROPERTIES
        BUILD_RPATH  "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )

    # Copy the three required OpenCV shared libraries after every build.
    foreach(_lib opencv_core opencv_imgproc opencv_tracking)
        add_custom_command(TARGET cvtracker POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${_lib}>
                $<TARGET_FILE_DIR:cvtracker>
            COMMENT "Bundling ${_lib} next to cvtracker"
        )
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Optional: install rules
# ---------------------------------------------------------------------------
install(TARGETS cvtracker
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/cvtracker_c.h DESTINATION include)
