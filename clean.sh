#!/bin/bash

clean_directory() {
  if [ -d "$1" ]; then
    echo "Removing directory: $1"
    rm -rf "$1"
  elif [ -f "$1" ]; then
    echo "Removing file: $1"
    rm -f "$1"
  else
    echo "$1 does not exist."
  fi
}

clean_build_system() {
  case "$1" in
    "build")
      clean_directory "build"
      ;;
    "cmake-build-debug")
      clean_directory "cmake-build-debug"
      ;;
    "Makefile")
      clean_directory "Makefile"
      ;;
    *)
      echo "Invalid argument: $1. Skipping."
      ;;
  esac
}

# Clean 'build', 'cmake-build-debug' directories, and 'Makefile'
clean_build_system "build"
clean_build_system "cmake-build-debug"
clean_build_system "Makefile"

echo "Clean-up successful."
