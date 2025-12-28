// tests/test_shape.cpp
// Unit tests for Shape class

#include <gtest/gtest.h>
#include <simd/simd.hpp>

using namespace micrograd::simd;

TEST(ShapeTest, SizeCalculation) {
    Shape s1 = {2, 3, 4};
    EXPECT_EQ(shape_size(s1), 24);
    
    Shape s2 = {100, 100};
    EXPECT_EQ(shape_size(s2), 10000);
    
    Shape empty = {};
    EXPECT_EQ(shape_size(empty), 0);
}

TEST(ShapeTest, StridesCalculation) {
    Shape s = {2, 3, 4};
    Strides strides = shape_strides(s);
    
    EXPECT_EQ(strides.size(), 3);
    EXPECT_EQ(strides[0], 12);  // 3 * 4
    EXPECT_EQ(strides[1], 4);   // 4
    EXPECT_EQ(strides[2], 1);   // 1
}

TEST(ShapeTest, RavelIndex) {
    Shape s = {2, 3};
    
    EXPECT_EQ(ravel_index({0, 0}, s), 0);
    EXPECT_EQ(ravel_index({0, 1}, s), 1);
    EXPECT_EQ(ravel_index({0, 2}, s), 2);
    EXPECT_EQ(ravel_index({1, 0}, s), 3);
    EXPECT_EQ(ravel_index({1, 2}, s), 5);
}

TEST(ShapeTest, ShapeToString) {
    Shape s1 = {2, 3, 4};
    EXPECT_EQ(shape_to_string(s1), "(2, 3, 4)");
    
    Shape s2 = {100};
    EXPECT_EQ(shape_to_string(s2), "(100)");
    
    Shape empty = {};
    EXPECT_EQ(shape_to_string(empty), "()");
}

