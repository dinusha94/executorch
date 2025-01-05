# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-src")
  file(MAKE_DIRECTORY "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-src")
endif()
file(MAKE_DIRECTORY
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-build"
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix"
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/tmp"
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp"
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src"
  "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/dinusha/simple_executorch/arm/build/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
