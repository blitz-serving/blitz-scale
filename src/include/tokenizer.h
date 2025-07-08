#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstdint>

namespace tokenizer {

struct Encoding;

struct Tokenizer;

extern "C" {

void free_encoding(Encoding *encoding_ptr);

uint32_t get_id(const Encoding *encoding, uintptr_t index);

uintptr_t num_ids(const Encoding *encoding);

void free_tokenizer(Tokenizer *tokenizer_ptr);

Tokenizer *create_tokenizer_from_file(const char *path);

Encoding *encode(Tokenizer *tokenizer, const char *input);

const char *decode(Tokenizer *tokenizer, const uint32_t *tokens, uintptr_t length);

void free_c_str_from_rust(const char *c_str);

} // extern "C"

} // namespace tokenizer

#endif // TOKENIZER_H
