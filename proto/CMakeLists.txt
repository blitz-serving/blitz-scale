#
# Protobuf/Grpc source files
#
set(PROTO_FILES generate.proto)

#
# Add Library target with protobuf sources
#
add_library(generatev2 SHARED ${PROTO_FILES})
target_link_libraries(generatev2
    PUBLIC
    protobuf::libprotobuf
    gRPC::grpc
    gRPC::grpc++
)
target_include_directories(generatev2 PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

#
# Compile protobuf and grpc files in myproto target to cpp
#
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET generatev2 LANGUAGE cpp)
protobuf_generate(TARGET generatev2 LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")
