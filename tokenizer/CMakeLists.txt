cmake_minimum_required(VERSION 3.18)

set(RUST_SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs)

add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/target/release/libtokenizer.a ${CMAKE_BINARY_DIR}/include/tokenizer.h
    COMMAND CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR} cargo build -r
    DEPENDS ${RUST_SOURCE_FILE}
    COMMENT "Running cargo build"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(rust_build
    DEPENDS ${CMAKE_SOURCE_DIR}/target/release/libtokenizer.a ${CMAKE_BINARY_DIR}/include/tokenizer.h
)

add_custom_target(check_rust_source
    ALL
    COMMAND ${CMAKE_COMMAND} -E touch ${RUST_SOURCE_FILE}
    DEPENDS ${RUST_SOURCE_FILE}
)

add_library(tokenizer STATIC IMPORTED GLOBAL)
add_dependencies(tokenizer rust_build)
set_target_properties(tokenizer
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/target/release/libtokenizer.a
)