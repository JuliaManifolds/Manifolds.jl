s = joinpath(@__DIR__, "..", "ManifoldsTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))
using ManifoldsTestSuite

using Manifolds, Test

M = Sphere(2)
p = [1.0, 0.0, 0.0]
q = [0.0, 1.0, 0.0]
X = [0.0, π / 2, 0.0]
Y = [0.0, 0.0, π / 2]

Manifolds.Test.test_manifold(
    M,
    Dict(
        :Functions => Manifolds.Test.all_functions(),
        :Points => [p, q], :Vectors => [X, Y]
    ),
    Dict(exp => q, manifold_dimension => 2)
);
