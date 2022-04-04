cmake_minimum_required(VERSION 3.10)
project(method1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(OpenCV 4.5 REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED)
find_package(nlohmann_json  REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        pcl_conversions
        pcl_ros
        cv_bridge
        )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)
set(SOURCES  src/common.h src/result_verify.h src/CustomMsg.h src/CustomPoint.h )

add_executable(getext1 ${SOURCES} src/cam_lid_external.cpp)
target_link_libraries(getext1 ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES} nlohmann_json::nlohmann_json)

add_executable(projectcloud ${SOURCES} src/projectCloud.cpp)
target_link_libraries(projectcloud ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES} nlohmann_json::nlohmann_json)