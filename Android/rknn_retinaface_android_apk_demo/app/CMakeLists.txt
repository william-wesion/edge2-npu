# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

# include rga.
include_directories(src/main/cpp/rga)

add_library( # Sets the name of the library.
             rknn4j

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cc
             src/main/cpp/post_process.cc
             src/main/cpp/retinaface_image.cc
             src/main/cpp/rknn_api.h
             src/main/cpp/object_tracker/track_link.cc
             src/main/cpp/object_tracker/objects_tracker.cc
             src/main/cpp/object_tracker/objects_update.cc
             )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       rknn4j

                       # Links the target library to the log library
                       # included in the NDK.
                       ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/librknnrt.so
                       ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/librga.so
                       ${log-lib} )