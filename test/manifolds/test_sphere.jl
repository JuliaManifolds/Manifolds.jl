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
        :InvalidPoints => [2 * p],
        :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
        :VectorTransportMethods => [ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport()]
    ),
    # Expectations
    Dict(
        :atol => 1.0e-12,
        distance => π / 4,
        exp => q,
        injectivity_radius => π,
        (injectivity_radius, ProjectionRetraction()) => π / 2,
        log => X, norm => π / 4,
        parallel_transport_to => parallel_transport_to(M, p, X, q),
        parallel_transport_direction => parallel_transport_to(M, p, X, q),
        manifold_dimension => 2,
        :IsPointErrors => [DomainError]
    ),
);
