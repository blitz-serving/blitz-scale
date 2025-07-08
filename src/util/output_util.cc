#include "output_util.h"

void writeLogToFile(const char* format, ...) {
    // 获取当前时间
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    // 打开日志文件
    FILE* logFile = fopen("/tmp/log.txt", "a");  // 以追加模式打开文件
    if (logFile == NULL) {
        fprintf(stderr, "Failed to open log file\n");
        return;
    }

    // 格式化日志消息
    char formattedMessage[1024];
    va_list args;
    va_start(args, format);
    int len =
        snprintf(formattedMessage, sizeof(formattedMessage), "[%ld.%09ld] ", ts.tv_sec, ts.tv_nsec);
    va_end(args);

    va_start(args, format);
    len += vsnprintf(formattedMessage + len, sizeof(formattedMessage) - len, format, args);
    va_end(args);

    // 写入日志文件
    fprintf(logFile, "%s\n", formattedMessage);

    // 关闭文件
    fclose(logFile);
}