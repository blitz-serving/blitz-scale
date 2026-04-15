use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/common.proto");
    println!("cargo:rerun-if-changed=../proto/engine.proto");
    println!("cargo:rerun-if-changed=../proto/resource.proto");
    println!("cargo:rerun-if-changed=../proto/admin.proto");

    fs::create_dir("src/pb").unwrap_or(());
    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src/pb")
        .include_file("mod.rs")
        .compile_with_config(
            prost_build::Config::new(),
            &[
                "../proto/common.proto",
                "../proto/engine.proto",
                "../proto/resource.proto",
                "../proto/admin.proto",
            ],
            &["../proto"],
        )
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));

    Ok(())
}
