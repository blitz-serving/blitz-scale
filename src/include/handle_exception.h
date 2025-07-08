#pragma once

#include "logger.hpp"

#define EXIT_ON_EXCEPTION                                                      \
  catch (const std::string &s) {                                               \
    BZ_FATAL(s);                                                               \
  }                                                                            \
  catch (const char *s) {                                                      \
    BZ_FATAL(s);                                                               \
  }                                                                            \
  catch (const std::exception &e) {                                            \
    BZ_FATAL(e.what());                                                        \
  }                                                                            \
  catch (...) {                                                                \
    BZ_FATAL("Unknown error");                                                 \
  }

#define UNWRAP EXIT_ON_EXCEPTION