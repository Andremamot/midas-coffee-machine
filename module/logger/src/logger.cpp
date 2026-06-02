#include <logger/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace {
std::shared_ptr<spdlog::logger> g_logger;
std::once_flag init_flag;

// Get current time string for log filename
std::string current_time_string() {
    auto t = std::time(nullptr);
    std::tm tm;

#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}
}  // namespace

namespace logger {

void init(const std::string& app_name) {
    std::call_once(init_flag, [&]() {
        // Create logs folder if needed
        std::string filename = "logs/" + app_name + "_" + current_time_string() + ".log";

        // File sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
        // Console sink (colored)
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
        g_logger = std::make_shared<spdlog::logger>("global_logger", sinks.begin(), sinks.end());

        g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
        g_logger->set_level(spdlog::level::debug);

        spdlog::set_default_logger(g_logger);
    });
}

void shutdown() { spdlog::shutdown(); }

void info(const std::string& msg) {
    if (g_logger) g_logger->info(msg);
}

void warn(const std::string& msg) {
    if (g_logger) g_logger->warn(msg);
}

void error(const std::string& msg) {
    if (g_logger) g_logger->error(msg);
}

void debug(const std::string& msg) {
    if (g_logger) g_logger->debug(msg);
}

}  // namespace logger