cmake_minimum_required(VERSION 3.16)
project(photon_MC LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

add_executable(
	photon_MC
	photon_MC.cpp	
)

target_compile_features(
	photon_MC
	PRIVATE cxx_std_17
)

target_link_libraries(
	photon_MC
	PRIVATE
	Eigen3::Eigen
	Threads::Threads
)

target_compile_options(
	photon_MC
	PRIVATE
	-Wall
	-Wextra
	-Wpedantic
)
