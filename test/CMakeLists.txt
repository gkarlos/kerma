message(STATUS "Building Polybench")
add_custom_command(
                OUTPUT Polybench_IR_Generated_Successfully
                DEPENDS kerma-make.sh
                WORKING_DIRECTORY ../examples/polybench/cuda/
                COMMAND touch Polybench_IR_Generated_Successfully
                COMMAND ./kerma-make.sh -ir -v
                )
add_custom_target(Polybench_IR_Gen DEPENDS Polybench_IR_Generated_Successfully)

message(STATUS "Building Polybench -- done")

add_subdirectory(unit)