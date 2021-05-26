const TEST_FLOAT32 = false
const TEST_DOUBLE64 = false
const TEST_STATIC_SIZED = false

using Manifolds
using ManifoldsBase
using ManifoldsBase: number_of_coordinates

using LinearAlgebra
using Distributions
using DoubleFloats
using ForwardDiff
using Quaternions
using Random
using ReverseDiff
using StaticArrays
using Statistics
using StatsBase
using Test
using LightGraphs
using SimpleWeightedGraphs

function include_test(path)
    @info "Testing $path"
    @time include(path)  # show basic timing, (this will print a newline at end)
end

function our_base_ambiguities()
    ambigs = Test.detect_ambiguities(Base)
    modules_we_care_about =
        [Base, LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Statistics, StatsBase]
    our_ambigs = filter(ambigs) do (m1, m2)
        we_care = m1.module in modules_we_care_about && m2.module in modules_we_care_about
        return we_care && (m1.module === Manifolds || m2.module === Manifolds)
    end
    return our_ambigs
end
