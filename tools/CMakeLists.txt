cmake_minimum_required(VERSION 3.5.1)

function(build_tool SRCFILE)
  get_filename_component(target ${SRCFILE} NAME_WE)
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    )
endfunction(build_tool)

build_tool(${PROJECT_SOURCE_DIR}/tools/VoiceActivityDetection-CTC.cpp)

# ---------------------- streaming_tds_model_converter ----------------------

add_executable(
  streaming_tds_model_converter
  ${CMAKE_CURRENT_LIST_DIR}/StreamingTDSModelConverter.cpp
)

target_include_directories(
  streaming_tds_model_converter
  PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)

target_link_libraries(
  streaming_tds_model_converter
  PRIVATE
    common
    criterion
    module
    runtime
    ${GFLAGS_LIBRARIES}
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
)

# ---------------------- saug add tool ----------------------

add_executable(
  add_saug
  ${CMAKE_CURRENT_LIST_DIR}/AddSAUG.cpp
)

target_include_directories(
  add_saug
  PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)

target_link_libraries(
  add_saug
  PRIVATE
    common
    criterion
    module
    runtime
    ${GFLAGS_LIBRARIES}
)

# ---------------------- streaming_tds_model_converter ----------------------

add_executable(
  streaming_recognize
  ${CMAKE_CURRENT_LIST_DIR}/StreamingRecognize.cpp
)

target_include_directories(
  streaming_recognize
  PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)

target_link_libraries(
  streaming_recognize
  PRIVATE
    common
    criterion
    module
    runtime
    ${GFLAGS_LIBRARIES}
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
    util_example
)