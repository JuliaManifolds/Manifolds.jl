TEST_FLOAT32 = get(ENV, "MANIFOLDS_TEST_FLOAT32", false)
TEST_DOUBLE64 = get(ENV, "MANIFOLDS_TEST_FLOAT64", false)
TEST_STATIC_SIZED = get(ENV, "MANIFOLDS_TEST_STATIC_SIZED", false)
TEST_GROUP = get(ENV, "MANIFOLDS_TEST_GROUP", "all")

using Manifolds
using ManifoldsBase
using ManifoldsBase: number_of_coordinates
import ManifoldsBase: active_traits, merge_traits

using ManifoldDiff

using LinearAlgebra
using Distributions
using DoubleFloats
using Quaternions
using Random
using StaticArrays
using Statistics
using StatsBase
using Test
using Graphs
using SimpleWeightedGraphs

function include_test(path)
    @info "Testing $path"
    @time include(path)  # show basic timing, (this will print a newline at end)
end

function our_ambiguities(m=Base)
    ambigs = Test.detect_ambiguities(m)
    modules_we_care_about =
        [Base, LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Statistics, StatsBase]
    our_ambigs = filter(ambigs) do (m1, m2)
        we_care = m1.module in modules_we_care_about && m2.module in modules_we_care_about
        return we_care && (m1.module === Manifolds || m2.module === Manifolds)
    end
    return our_ambigs
end

"""
    has_type_in_signature(sig, T)
    Test whether the signature `sig` has an argument of type `T` as one of its paramaters
"""
function has_type_in_signature(sig, T::Type)
    return any(map(Base.unwrap_unionall(sig.sig).parameters) do x
        xw = Base.rewrap_unionall(x, sig.sig)
        return (xw isa Type ? xw : xw.T) <: T
    end)
end
