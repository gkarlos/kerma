#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
kermaRoot="$(dirname "$DIR")"

if [ ! -d ${kermaRoot} ]; then
  echo "Doxygen docs not built"
  exit
fi

if which xdg-open > /dev/null
then
  xdg-open ${kermaRoot}/build/docs/html/index.html
elif which gnome-open > /dev/null
then
  gnome-open ${kermaRoot}/build/docs/html/index.html
fi