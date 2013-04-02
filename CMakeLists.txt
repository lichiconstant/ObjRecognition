cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ObjRecognition)

find_package(OpenCV REQUIRED )
find_package(PCL 1.6 REQUIRED)
find_package(OpenMP)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(feature_srcs
	features.cpp	
	)

set(feature_includes
	features.h
	typedefs.h
	)

set(toro3d_srcs
	posegraph3.cpp
	treeoptimizer3.cpp
	treeoptimizer3_iteration.cpp
	)

set(toro3d_includes
	dmatrix.h
	dmatrix.hxx
	posegraph.h
	transformation3.h
	transformation3.hxx
	treeoptimizer3.h
	)

add_library(Features ${feature_srcs} ${feature_includes})
target_link_libraries (Features ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_library(Recognizer Recognizer.h VotingRecog.cpp SegRecog.cpp)
target_link_libraries (Recognizer Features ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_library(Model_Builder Model_Builder.h Model_Builder.cpp)
target_link_libraries (Model_Builder ${PCL_LIBRARIES} ${OpenCV_LIBS})

#add_library(Toro3d ${toro3d_srcs} ${toro3d_includes})

add_executable(Training Training.cpp)
target_link_libraries(Training ${PCL_LIBRARIES} ${OpenCV_LIBS} Features)

add_executable(seg_recog seg_recog.cpp)
target_link_libraries(seg_recog ${PCL_LIBRARIES} ${OpenCV_LIBS} Features Recognizer)

add_executable(Preprocess preprocess.cpp)
target_link_libraries(Preprocess ${PCL_LIBRARIES} ${OpenCV_LIBS} Features)

add_executable(CropModel CropModel.cpp)
target_link_libraries(CropModel ${PCL_LIBRARIES} ${OpenCV_LIBS} Features)

add_executable(ViewModel ViewModel.cpp)
target_link_libraries(ViewModel ${PCL_LIBRARIES} ${OpenCV_LIBS} Features)

add_executable(FilterScene FilterScene.cpp)
target_link_libraries(FilterScene ${PCL_LIBRARIES} ${OpenCV_LIBS} Features)

add_executable(Test test.cpp)
target_link_libraries(Test ${PCL_LIBRARIES} ${OpenCV_LIBS} Features Model_Builder)

add_executable(vote_recog vote_recog.cpp)
target_link_libraries(vote_recog ${PCL_LIBRARIES} ${OpenCV_LIBS} Features Recognizer)

add_executable(BuildModel BuildModel.cpp)
target_link_libraries(BuildModel ${PCL_LIBRARIES} ${OpenCV_LIBS} Features Model_Builder)

