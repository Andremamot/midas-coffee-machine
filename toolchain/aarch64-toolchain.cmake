set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Compilers
set(CMAKE_C_COMPILER aarch64-poky-linux-gcc)
set(CMAKE_CXX_COMPILER aarch64-poky-linux-g++)

# Sysroot
set(CMAKE_SYSROOT /opt/poky/3.1.31/sysroots/aarch64-poky-linux)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Flags
set(CMAKE_C_FLAGS "--sysroot=${CMAKE_SYSROOT} -mtune=cortex-a55 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -Wformat -Wformat-security -Werror=format-security")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17")
