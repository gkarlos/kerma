add_custom_target(BuildPolybenchIR
    COMMAND 
)

message(STATUS "Configuring Polybench")

add_custom_target( PolybenchGenIR ALL)
add_custom_command( TARGET PolybenchGenIR
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMAND ./kerma-make.sh -ir -silent)

message(STATUS "Configuring Polybench -- done")


# set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "*.ll")