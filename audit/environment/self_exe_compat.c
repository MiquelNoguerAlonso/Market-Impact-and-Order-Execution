#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
/* Lean 4.19 asks for /proc/<namespace pid>/exe. This environment exposes
   the calling executable at /proc/self/exe. Use only that permitted self
   reference; do not access or change permissions on any numbered process. */
ssize_t readlink(const char *path, char *buf, size_t size) {
    static ssize_t (*real_readlink)(const char *, char *, size_t);
    if (!real_readlink) real_readlink = dlsym(RTLD_NEXT, "readlink");
    char own_path[80];
    snprintf(own_path, sizeof own_path, "/proc/%ld/exe", (long)getpid());
    if (strcmp(path, own_path) == 0) path = "/proc/self/exe";
    return real_readlink(path, buf, size);
}
