include(ExternalProject)

set(CUB_URL https://github.com/NVlabs/cub.git)
set(CUB_TAG c3cceac115c072fb63df1836ff46d8c60d9eb304) # release 1.8.0

# Download CUB
if (CUB_ROOT_DIR)
  ExternalProject_Add(
    CUB
    PREFIX CUB
    SOURCE_DIR ${CUB_ROOT_DIR}
    BUILD_IN_SOURCE 0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
else()
  ExternalProject_Add(
    CUB
    PREFIX CUB
    GIT_REPOSITORY ${CUB_URL}
    GIT_TAG ${CUB_TAG}
    BUILD_IN_SOURCE 0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
endif()
ExternalProject_Get_Property(CUB source_dir)
set(CUB_SOURCE_DIR ${source_dir})
ExternalProject_Get_Property(CUB binary_dir)
set(CUB_BINARY_DIR ${binary_dir})

# Include dir. No install step supported yet, so invariant/no gen exps needed
set(CUB_INCLUDE_DIRS ${CUB_SOURCE_DIR})
