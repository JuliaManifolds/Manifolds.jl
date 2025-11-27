using Manifolds, ManifoldsBase, Test, StaticArrays

# This file tests a few generic functions for general unitary matricesd
# independent of their matrix type, so we check that with a dummy type

Test.@testset "GeneralUnitaryMatrices" begin
    M2 = Manifolds.GeneralUnitaryMatrices(2, ℝ, Manifolds.Test.DummyMatrixType)
    M2f = Manifolds.GeneralUnitaryMatrices(2, ℝ, Manifolds.Test.DummyMatrixType; parameter = :field)
    @testset "Basics" begin
        @test base_manifold(M2) === get_embedding(M2)
    end
    p2 = [1.0 0.0; 0.0 1.0]
    X2 = [0.0 -1.0; 1.0 0.0]
    p2S = @SMatrix [1.0 0.0; 0.0 1.0]
    X2S = @SMatrix [0.0 -1.0; 1.0 0.0]
    c2 = [1.0]
    b = DefaultOrthogonalBasis()

    for M in [M2, M2f], (p, X) in zip([p2, p2S], [X2, X2S])
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [get_coordinates, get_vector], :Bases => [b],
                :Points => [p], :Vectors => [X], :Coordinates => [c2],
            ),
            Dict((get_coordinates, b) => c2, (get_vector, c2, b) => X)
        )
    end


    M3 = Manifolds.GeneralUnitaryMatrices(3, ℝ, Manifolds.Test.DummyMatrixType)
    M3f = Manifolds.GeneralUnitaryMatrices(3, ℝ, Manifolds.Test.DummyMatrixType; parameter = :field)

    p3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    X3 = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    p3S = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    X3S = @SMatrix [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    c3 = [0.0, 0.0, 1.0]

    for M in [M3f, M3], (p, X) in zip([p3, p3S], [X3, X3S])
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [get_coordinates, get_vector], :Bases => [b],
                :Points => [p], :Vectors => [X], :Coordinates => [c3],
            ),
            Dict((get_coordinates, b) => c3, (get_vector, c3, b) => X)
        )
    end
    @testset "Stattic Array & Edge cases" begin
        Z = @SMatrix [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        q = exp(M3, p3S, Z)
        @test q == p3S
        q3S = @SMatrix [-1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]
        X = log(M3, p3S, q3S)
        X3S = @SMatrix [0.0 -π 0.0; π 0.0 0.0; 0.0 0.0 0.0]
        @test isapprox(M3, X, X3S)
    end


    M4 = Manifolds.GeneralUnitaryMatrices(4, ℝ, Manifolds.Test.DummyMatrixType)
    M4f = Manifolds.GeneralUnitaryMatrices(4, ℝ, Manifolds.Test.DummyMatrixType; parameter = :field)

    p4 = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    X4 = [0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    c4 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for M in [M4, M4f]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Functions => [get_coordinates, get_vector], :Bases => [b],
                :Points => [p4], :Vectors => [X4], :Coordinates => [c4],
            ),
            Dict((get_coordinates, b) => c4, (get_vector, c4, b) => X4)
        )
    end
end
