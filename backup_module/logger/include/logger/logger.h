#pragma once

#include <memory>
#include <string>

namespace logger {

void init(const std::string& app_name = "app");
void shutdown();

void info(const std::string& msg);
void error(const std::string& msg);
void warn(const std::string& msg);
void debug(const std::string& msg);

}  // namespace logger

// =====================
// MACRO CONTROL (__DEBUG)
// =====================
#ifdef __DEBUG

#define LOGR_INIT(app) logger::init(app)
#define LOGR_SHUTDOWN() logger::shutdown()

#define LOGR_INFO(msg) logger::info(msg)
#define LOGR_ERROR(msg) logger::error(msg)
#define LOGR_WARN(msg) logger::warn(msg)
#define LOGR_DEBUG(msg) logger::debug(msg)

#else

#define LOGR_INIT(app)
#define LOGR_SHUTDOWN()

#define LOGR_INFO(msg)
#define LOGR_ERROR(msg)
#define LOGR_WARN(msg)
#define LOGR_DEBUG(msg)

#endif