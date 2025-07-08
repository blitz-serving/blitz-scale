#pragma once

#include <cassert>
#include <iostream>

namespace blitz {
using rank_t = int;

/**
 *  \brief An shallow embedding for Assertion Logic
 *
 *  \example flags 
 */

constexpr bool None = true;
constexpr bool Verbose = false;
constexpr bool Cliche = false;
constexpr bool All = false;
static_assert(not (None && All), "Check compilation option!");

namespace flag {
constexpr bool Init = false && !None;
constexpr bool Cuda = true && !None;
constexpr bool Rdma = false && !None;
constexpr bool Param = false && !None;
constexpr bool Tanz = false && !None;
constexpr bool Zigzag = true && !None;
}  // namespace flag

#define Assertion(expr, flag)                                                 \
    do {                                                                      \
        if constexpr (flag) {                                                 \
            if (!(expr)) {                                                    \
                std::cerr << "Invalid assertion logic: " << #expr << "("      \
                          << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
                std::abort();                                                 \
            }                                                                 \
        }                                                                     \
    } while (0)

#define Precondition(expr, flag)                                              \
    do {                                                                      \
        if constexpr (flag) {                                                 \
            if (!(expr)) {                                                    \
                std::cerr << "Invalid assertion logic: " << #expr << "("      \
                          << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
                std::abort();                                                 \
            }                                                                 \
        }                                                                     \
    } while (0)

#define Postcondition(expr, flag)                                             \
    do {                                                                      \
        if constexpr (flag) {                                                 \
            if (!(expr)) {                                                    \
                std::cerr << "Invalid assertion logic: " << #expr << "("      \
                          << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
                std::abort();                                                 \
            }                                                                 \
        }                                                                     \
    } while (0)

#define Invariant(expr, flag)                                                 \
    do {                                                                      \
        if constexpr (flag) {                                                 \
            if (!(expr)) {                                                    \
                std::cerr << "Unsatisfied invariant: " << #expr << "("        \
                          << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
                std::abort();                                                 \
            }                                                                 \
        }                                                                     \
    } while (0)

}  // namespace blitz