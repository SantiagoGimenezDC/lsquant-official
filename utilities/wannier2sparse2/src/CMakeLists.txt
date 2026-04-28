# @author  Santiago Gimenez de Castro

set(LIB_HEADERS
)

set(LIB_SOURCES
)

set(APP_HEADERS
)

set(APP_SOURCES
  #wannier_velocity_to_csr.cpp
  wannier_hr_to_csr.cpp
)
# Set the build version. It will be used in the name of the lib, with corresponding
# symlinks created. SOVERSION could also be specified for api version. 
set_target_properties(${LIBRARY_NAME} PROPERTIES
  VERSION ${VERSION}  # ${VERSION} was defined in the main CMakeLists.
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}" # specify the public headers
)

# This is a make target, so you can do a "make projArunner"
set(APPLICATION_NAME wannier2sparse2)
add_executable(${APPLICATION_NAME} ${APP_SOURCES})

