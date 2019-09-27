#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <iostream>
#include <string>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include "color.h"

namespace std {
string to_string(string s) __attribute__((weak));
string to_string(void* p) __attribute__((weak));
string to_string(char* s) __attribute__((weak));
string to_string(const char* s) __attribute__((weak));
string to_string(bool b) __attribute__((weak));
template <typename A, typename B>
string to_string(std::pair<A, B> p) __attribute__((weak));
template <typename A>
string to_string(A v) __attribute__((weak));

string to_string(string s) { return '"' + s + '"'; }

string to_string(void* p) {
  char address[20];
  sprintf(address, "%p", p);
  return string(address);
}

string to_string(char* s) { return to_string((string)s); }

string to_string(const char* s) { return to_string((string)s); }

string to_string(bool b) { return (b ? "true" : "false"); }
template <typename A, typename B>
string to_string(std::pair<A, B> p) {
  return "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
}

template <typename A>
string to_string(A v) {
  bool first = true;
  string res = "{";
  for (const auto& x : v) {
    if (!first) {
      res += ", ";
    }
    first = false;
    res += to_string(x);
  }
  res += "}";
  return res;
}

}  // namespace std

void debug_out() __attribute__((weak));
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) __attribute__((weak));

void debug_out() { std::cerr << COLOR_END << std::endl; }
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) {
  std::cerr << RED << " " << std::to_string(H);
  debug_out(T...);
}

#ifdef DEBUG
#define debug(...) \
  std::cerr << "[" << #__VA_ARGS__ << "]:", debug_out(__VA_ARGS__)
#else
#define debug(...) 42
#endif

#endif
