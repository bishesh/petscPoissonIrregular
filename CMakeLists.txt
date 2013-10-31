cmake_minimum_required(VERSION 2.8)
project(poissonIrregular)

#set(PETSC_DIR /home/bkhanal/Documents/softwares/petsc/)
set(PETSC_DIR /home/bkhanal/Documents/softwares/petsc-3.4.3/)
set(PETSC_ARCH arch-linux2-cxx-debug)

#add cmake modules path to the default one:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/user/bkhanal/home/cmakeModules/petsc/")

#Find the external library for Petsc
find_package(PETSc COMPONENTS CXX REQUIRED)
set(CMAKE_CXX_COMPILER ${PETSC_COMPILER})

include_directories(${PETSC_INCLUDES})
link_directories(${PETSC_LIBRARIES})


add_executable(poissonIrregular poissonIrregular.cxx)

target_link_libraries(poissonIrregular ${PETSC_LIBRARIES})


