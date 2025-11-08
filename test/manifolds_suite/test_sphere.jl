s = joinpath(@__DIR__, "..", "ManifoldsTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))
using ManifoldsTestSuite

using Manifolds, Test

M = Sphere(2)
p = [1.0, 0.0, 0.0]
q = [1 / sqrt(2), 1 / sqrt(2), 0.0]
X = [0.0, π / 4, 0.0]
Y = [0.0, 0.0, π / 4]

Manifolds.Test.test_manifold(
    M,
    Dict(
        :Functions => Manifolds.Test.all_functions(),
        :InverseRetractionMethods => [LogarithmicInverseRetraction(), ProjectionInverseRetraction()],
        :Points => [p, q], :Vectors => [X, Y],
        :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
        :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
    ),
    # Expectations
    Dict(
        :atol => 1.0e-12,
        exp => q,
        manifold_dimension => 2
    ),
);
