using Manifolds
using ManifoldsBase
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

"""
    has_type_in_signature(sig, T::Type)

Test whether the signature `sig` has an argument of type `T` as one of its parameters.
"""
function has_type_in_signature(sig, T::Type)
    return any(
        map(Base.unwrap_unionall(sig.sig).parameters) do x
            xw = Base.rewrap_unionall(x, sig.sig)
            return (xw isa Type ? xw : xw.T) <: T
        end
    )
end

@testset "Ambiguities" begin
    if VERSION.prerelease == () && !Sys.iswindows() && VERSION < v"1.11.0"
        mbs = Test.detect_ambiguities(ManifoldsBase)
        # Interims solution until we follow what was proposed in
        # https://discourse.julialang.org/t/avoid-ambiguities-with-individual-number-element-identity/62465/2
        MBS_LIMIT = 41
        println("Number of ManifoldsBase.jl ambiguities: $(length(mbs))")
        @test length(mbs) <= MBS_LIMIT
        if length(mbs) > MBS_LIMIT
            for amb in mbs
                println(amb)
                println()
            end
        end
        ms = Test.detect_ambiguities(Manifolds)
        # Interims solution until we follow what was proposed in
        # https://discourse.julialang.org/t/avoid-ambiguities-with-individual-number-element-identity/62465/2
        MS_LIMIT = 46
        println("Number of Manifolds.jl ambiguities: $(length(ms))")
        if length(ms) > MS_LIMIT
            for amb in ms
                println(amb)
                println()
            end
        end
        @test length(ms) <= MS_LIMIT
        # this test takes way too long to perform regularly
        # @test length(our_base_ambiguities()) <= 4
    else
        @info "Skipping Ambiguity tests for pre-release versions"
    end
end
