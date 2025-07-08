extern crate cbindgen;

use std::{env, fs, path};

fn main() {
    println!("cargo:rerun-if-changed=./src/lib.rs");
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    match env::var("CMAKE_BINARY_DIR") {
        Ok(cmake_binary_dir) => {
            let output_dir = path::PathBuf::new().join(cmake_binary_dir).join("include");
            fs::create_dir_all(&output_dir).expect("Failed to create output directory");
            cbindgen::Builder::new()
                .with_language(cbindgen::Language::Cxx)
                .with_namespace("tokenizer")
                .with_include_guard("TOKENIZER_H")
                .with_sys_include("cstdint")
                .with_no_includes()
                .with_crate(crate_dir)
                .generate()
                .expect("Failed to generate bindings")
                .write_to_file(output_dir.join("tokenizer.h"));
        }
        Err(_) => {
            eprintln!("CMAKE_BINARY_DIR is not set");
            return;
        }
    }
}
