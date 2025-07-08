#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer.h"

namespace blitz {

class HuggingfaceTokenizer {
  private:
    ::tokenizer::Tokenizer* tokenizer_ptr;

  public:
    HuggingfaceTokenizer() = delete;
    explicit HuggingfaceTokenizer(const std::string& tokenizer_json_path) {
        tokenizer_ptr = tokenizer::create_tokenizer_from_file(tokenizer_json_path.c_str());
    }
    ~HuggingfaceTokenizer() {
        tokenizer::free_tokenizer(tokenizer_ptr);
    }

    std::vector<uint32_t> encode(const std::string& text) const {
        tokenizer::Encoding* encoding_ptr = tokenizer::encode(tokenizer_ptr, text.c_str());
        std::vector<uint32_t> ids;
        if (encoding_ptr != nullptr) {
            auto num_tokens = tokenizer::num_ids(encoding_ptr);
            for (size_t i = 0; i < num_tokens; ++i) {
                ids.push_back(tokenizer::get_id(encoding_ptr, i));
            }
        } else {
            throw std::runtime_error("HFTokenizer encode error!");
        }
        tokenizer::free_encoding(encoding_ptr);
        return ids;
    }

    std::string decode(uint32_t token) const {
        const char* text_ptr = tokenizer::decode(tokenizer_ptr, &token, 1);
        std::string text(text_ptr);
        tokenizer::free_c_str_from_rust(text_ptr);
        return text;
    }

    std::string decode(const std::vector<uint32_t>& tokens) const {
        // std::string can be converted from C-string
        const char* text_ptr = tokenizer::decode(tokenizer_ptr, tokens.data(), tokens.size());
        // copy null-terminated char from Rust realm to C++
        std::string text(text_ptr);
        tokenizer::free_c_str_from_rust(text_ptr);
        return text;
    }
};

} // namespace blitz


