# Header
using Pkg; Pkg.activate(@__DIR__)
if VERSION < v"1.11"
    s = joinpath(@__DIR__, "ManifoldsTestSuite.jl")
    !(s in LOAD_PATH) && (push!(LOAD_PATH, s))
    Pkg.resolve()
end
#
using ManifoldsTestSuite
using Manifolds, LinearAlgebra, Test, Random
