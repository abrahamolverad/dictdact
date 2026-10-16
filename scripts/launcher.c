// DictDact.app main executable: a tiny Mach-O that execs the bundled CPython with `-m dictdact run`.
// (A shell-script main executable cannot carry a hardened-runtime signature, which notarization needs.)
#include <limits.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char exe[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0) {
        fprintf(stderr, "dictdact: executable path too long\n");
        return 1;
    }
    char real[PATH_MAX];
    if (realpath(exe, real) == NULL) {
        perror("dictdact: realpath");
        return 1;
    }
    char *dir = dirname(real); /* .../DictDact.app/Contents/MacOS */
    char python[PATH_MAX];
    snprintf(python, sizeof(python), "%s/python", dir);

    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONIOENCODING", "utf-8", 1);
    setenv("DICTDACT_BUNDLED", "1", 1);
    const char *path = getenv("PATH");
    char newpath[8192];
    snprintf(newpath, sizeof(newpath), "/opt/homebrew/bin:/usr/local/bin:%s", path ? path : "/usr/bin:/bin");
    setenv("PATH", newpath, 1);

    /* python -m dictdact run [extra args...] */
    char **args = calloc((size_t)argc + 5, sizeof(char *));
    int n = 0;
    args[n++] = python;
    args[n++] = "-m";
    args[n++] = "dictdact";
    if (argc > 1) {
        for (int i = 1; i < argc; i++) args[n++] = argv[i];
    } else {
        args[n++] = "run";
    }
    args[n] = NULL;
    execv(python, args);
    perror("dictdact: execv");
    return 1;
}
