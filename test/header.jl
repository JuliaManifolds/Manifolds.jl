TEST_FLOAT32 = get(ENV, "MANIFOLDS_TEST_FLOAT32", false)
TEST_DOUBLE64 = get(ENV, "MANIFOLDS_TEST_FLOAT64", false)
TEST_STATIC_SIZED = get(ENV, "MANIFOLDS_TEST_STATIC_SIZED", false)
TEST_GROUP = get(ENV, "MANIFOLDS_TEST_GROUP", "all")

using Manifolds
using ManifoldsBase
using ManifoldsBase: number_of_coordinates, TypeParameter

using ManifoldDiff

using LinearAlgebra
using Distributions
using DoubleFloats
using Quaternions
using Random
using StaticArrays
using RecursiveArrayTools
using Statistics
using StatsBase
using Test
using Graphs
using SimpleWeightedGraphs
