#!/bin/bash
BOOST_VERSION="1_72_0"
BOOST_ROOT="${HOME}/boost/172"
BOOST_INC="${BOOST_ROOT}/include"
BOOST_LIB="${BOOST_ROOT}/lib"
BOOST_LIBRARIES=(system filesystem chrono atomic date_time)

function missingLibs {
  for LIB in $BOOST_LIBRARIES; do
    if [[ ! -e ~/$BOOST_LIB/"libboost_${LIB}.so" ]]; then
      return 0;
    fi;
  done
  return 1
}

function join () {
  local IFS="$1"
  shift
  echo "$*"
}

if missingLibs; then
  cd ~/
  wget "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_${BOOST_VERSION}.tar.bz2"
  tar --bzip2 -xf "./boost_${BOOST_VERSION}.tar.bz2"
  cd "boost_${BOOST_VERSION}"
  WITH_LIBS=`join , "${BOOST_LIBRARIES[@]}"`
  ./bootstrap.sh --prefix=$HOME/boost/172 variant=release address-model=64 cxxflags="-std=c++14 -fPIC"\
    --libdir=$BOOST_LIB --includedir=$BOOST_INC --with-libraries=$WITH_LIBS
  ./b2 install
fi