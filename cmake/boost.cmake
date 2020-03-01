message(STATUS "Detecting Boost")

set(KERMA_BOOST_MIN_VERSION "1.66.0")

# when adding components remember to update scripts/travis/install_boost.sh too for CI
set(KERMA_BOOST_COMPONENTS system filesystem)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIM OFF)

# If $BOOST_HOME or $BOOST_ROOT is set then use that directory as BOOST_ROOT
# and expect directories "lib" and "inc" under it
# If neither $BOOST_HOME nor $BOOST_ROOT is set then check for default installation

if(DEFINED ENV{BOOST_ROOT} OR DEFINED ENV{BOOST_HOME})
    set( Boost_NO_SYSTEM_PATHS on CACHE BOOL "Do not search system for Boost" )

    if(DEFINED ENV{BOOST_ROOT})
        set(BOOST_ROOT $ENV{BOOST_ROOT} CACHE PATH "Boost library path")
        message("   [-]  Found $BOOST_ROOT = ${BOOST_ROOT}")
    else()
        set(BOOST_ROOT $ENV{BOOST_HOME} CACHE PATH "Boost library path")
        message("   [-]  Found $BOOST_HOME = ${BOOST_ROOT}")
    endif()
else()
    message("   [-]  Checking default system installation")
endif()

find_package(Boost)
if (Boost_FOUND)
    message("        Found INC: ${Boost_INCLUDE_DIRS}")
    # TODO CI prints nothing for ${Boost_LIBRARY_DIRS} even though it reports Boost found
    message("        Found LIB: ${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

message(STATUS "Detecting Boost - done")