#include <gtest/gtest.h>
#include <pipepye/utils/timer.hpp>
#include <thread>
#include <chrono>

TEST(TimerTest, MeasureElapsedDuration) {
    pipepye::utils::CPUTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.stop();

    double ms = timer.elapsed_milliseconds();
    // Verify measured sleep is at least 15ms and within reasonable bounds
    EXPECT_GE(ms, 15.0);
    EXPECT_LE(ms, 100.0);

    double sec = timer.elapsed_seconds();
    EXPECT_GE(sec, 0.015);
    EXPECT_LE(sec, 0.10);
}

TEST(TimerTest, ResetFunctionality) {
    pipepye::utils::CPUTimer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();
    EXPECT_GT(timer.elapsed_milliseconds(), 5.0);

    timer.reset();
    EXPECT_DOUBLE_EQ(timer.elapsed_milliseconds(), 0.0);
}
