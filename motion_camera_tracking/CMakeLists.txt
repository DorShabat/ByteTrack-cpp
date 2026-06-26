cmake_minimum_required(VERSION 3.16)

project(MotionEstimator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d)

add_library(MotionEstimator SHARED
    src/motion_estimator_c_api.cpp
)

target_include_directories(MotionEstimator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(MotionEstimator
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_video
        opencv_calib3d
)

target_compile_definitions(MotionEstimator
    PRIVATE
        MOTION_ESTIMATOR_EXPORTS
)

set_target_properties(MotionEstimator PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------------------
# Bundling: on Windows and Linux copy OpenCV libraries next to the output.
# On Apple the Homebrew install is used as-is (no bundling needed).
# ---------------------------------------------------------------------------
if(NOT APPLE)
    set_target_properties(MotionEstimator PROPERTIES
        BUILD_RPATH   "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )

    foreach(_lib opencv_core opencv_imgproc opencv_video opencv_calib3d)
        add_custom_command(TARGET MotionEstimator POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:${_lib}>
                $<TARGET_FILE_DIR:MotionEstimator>
            COMMENT "Bundling ${_lib} next to MotionEstimator"
        )
    endforeach()
endif()