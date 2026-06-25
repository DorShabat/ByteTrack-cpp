cmake_minimum_required(VERSION 3.14)
project(bytetrack)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

# Check C++17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  add_definitions(-DCOMPILEDWITHC17)
  message(STATUS "Using flag -std=c++17.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

include(FetchContent)

set(EIGEN_VERSION 3.3)
set(EIGEN_FETCH_VERSION 3.4.0)
set(BYTETRACK_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" CACHE PATH "Path to bundled Eigen directory")

if(EXISTS "${BYTETRACK_EIGEN_DIR}/Eigen/Core")
  message(STATUS "Using bundled Eigen from: ${BYTETRACK_EIGEN_DIR}")
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${BYTETRACK_EIGEN_DIR}")
  endif()
else()
  find_package(Eigen3 ${EIGEN_VERSION} QUIET)
  if(NOT EIGEN3_FOUND)
    set(BUILD_TESTING OFF CACHE INTERNAL "")
    FetchContent_Declare(eigen
      GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
      GIT_TAG         ${EIGEN_FETCH_VERSION}
      GIT_SHALLOW     ON)
    FetchContent_MakeAvailable(eigen)
    unset(BUILD_TESTING CACHE)
  endif()
endif()

add_library(${PROJECT_NAME} SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/BYTETracker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KalmanFilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bytetrack_c.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lapjv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Object.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Rect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/STrack.cpp
  )
target_compile_definitions(${PROJECT_NAME} PRIVATE BYTETRACK_EXPORTS)
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# Build the tests if the 'BUILD_BYTETRACK_TEST' variable is set to 'ON'
set(BUILD_BYTETRACK_TEST OFF CACHE BOOL "The flag whether to build the tests or not")
if(BUILD_BYTETRACK_TEST)
  enable_testing()
  include(GoogleTest)

  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    message(FATAL_ERROR "GTest not found.")
  endif()

  add_executable(${PROJECT_NAME}_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_BYTETracker.cpp
  )

  target_link_libraries(${PROJECT_NAME}_test
    ${PROJECT_NAME}
    GTest::GTest
    GTest::Main
    )

  gtest_discover_tests(${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/YOLOX_ncnn_palace)
endif()
