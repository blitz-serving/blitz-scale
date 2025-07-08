mod bindings {
    use std::ffi::{c_char, CStr, CString};

    #[repr(align(8))]
    pub struct Encoding;

    #[no_mangle]
    pub extern "C" fn free_encoding(encoding_ptr: *mut Encoding) {
        if !encoding_ptr.is_null() {
            let _encoding = unsafe { Box::from_raw(encoding_ptr as *mut tokenizers::Encoding) };
        }
    }

    #[no_mangle]
    pub extern "C" fn get_id(encoding: *const Encoding, index: usize) -> u32 {
        unsafe {
            (encoding as *const tokenizers::Encoding)
                .as_ref()
                .unwrap()
                .get_ids()[index]
        }
    }

    #[no_mangle]
    pub extern "C" fn num_ids(encoding: *const Encoding) -> usize {
        unsafe {
            (encoding as *const tokenizers::Encoding)
                .as_ref()
                .unwrap()
                .get_ids()
                .len()
        }
    }

    #[repr(align(8))]
    pub struct Tokenizer;

    #[no_mangle]
    pub extern "C" fn free_tokenizer(tokenizer_ptr: *mut Tokenizer) {
        if !tokenizer_ptr.is_null() {
            let _tokenizer = unsafe { Box::from_raw(tokenizer_ptr as *mut tokenizers::Tokenizer) };
        }
    }

    #[no_mangle]
    pub extern "C" fn create_tokenizer_from_file(path: *const c_char) -> *mut Tokenizer {
        unsafe {
            let path = cstr_to_string(path);
            match tokenizers::Tokenizer::from_file(path.as_str()) {
                Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)) as _,
                Err(e) => {
                    println!(
                        "{}:{} Failed to create tokenizer from file {}. Error message: {}",
                        file!(),
                        line!(),
                        path,
                        e.to_string()
                    );
                    std::ptr::null_mut()
                }
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn encode(tokenizer: *mut Tokenizer, input: *const c_char) -> *mut Encoding {
        unsafe {
            match (tokenizer as *mut tokenizers::Tokenizer)
                .as_ref()
                .unwrap()
                .encode(cstr_to_string(input), true)
            {
                Ok(encoding) => Box::into_raw(Box::new(encoding)) as *mut Encoding,
                Err(e) => {
                    println!("{}:{} Error: {}", file!(), line!(), e.to_string());
                    std::ptr::null_mut()
                }
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn decode(
        tokenizer: *mut Tokenizer,
        tokens: *const u32,
        length: usize,
    ) -> *const c_char {
        unsafe {
            match (tokenizer as *mut tokenizers::Tokenizer)
                .as_ref()
                .unwrap()
                .decode(std::slice::from_raw_parts(tokens, length), false)
            {
                Ok(output) => CString::new(output).unwrap().into_raw(),
                Err(e) => {
                    println!("{}:{} Error: {}", file!(), line!(), e.to_string());
                    std::ptr::null()
                }
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn free_c_str_from_rust(c_str: *const c_char) {
        if !c_str.is_null() {
            drop(unsafe { CString::from_raw(c_str as *mut c_char) });
        }
    }

    pub unsafe fn cstr_to_string(cstr: *const c_char) -> String {
        CStr::from_ptr(cstr)
            .to_str()
            .expect("Failed to convert cstr to rust string")
            .to_owned()
    }
}
