using LinearAlgebra, Manifolds, Quaternions, Test, Random

@testset "Orthogonal Matrices" begin
    M = OrthogonalMatrices(3)
    α = 0.1
    p1 = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    p2 = [-cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    p3 = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
    pE = ones(3, 3)
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                default_vector_transport_method,
                get_embedding,
                injectivity_radius, is_flat,
                rand, repr,
            ],
            :EmbeddedPoints => [p3],
            :Points => [p1, p2, p3],
            :RetractionMethods => [PolarRetraction()],
        ),
        Dict(
            default_vector_transport_method => ProjectionTransport(),
            get_embedding => Euclidean(3, 3),
            injectivity_radius => π * sqrt(2),
            (injectivity_radius, PolarRetraction()) => π / sqrt(2),
            is_flat => false,
            manifold_dimension => 3,
            repr => "OrthogonalMatrices(3)",
        )
    )

    @testset "volume" begin
        @test manifold_volume(OrthogonalMatrices(1)) ≈ 2
        @test manifold_volume(OrthogonalMatrices(2)) ≈ 4 * π * sqrt(2)
        @test manifold_volume(OrthogonalMatrices(3)) ≈ 16 * π^2 * sqrt(2)
        @test manifold_volume(OrthogonalMatrices(4)) ≈ 2 * (2 * π)^4 * sqrt(2)
        @test manifold_volume(OrthogonalMatrices(5)) ≈ 8 * (2 * π)^6 / 6 * sqrt(2)
    end

    @testset "Field parameter" begin
        @test get_embedding(OrthogonalMatrices(3; parameter = :field)) === Euclidean(3, 3; parameter = :field)
        @test repr(OrthogonalMatrices(3; parameter = :field)) ==
            "OrthogonalMatrices(3; parameter=:field)"
    end

    @testset "Selected methods on OrthogonalMatrices(1)" begin
        @test abs(rand(OrthogonalMatrices(1))[]) == 1
        @test abs(rand(MersenneTwister(), OrthogonalMatrices(1))[]) == 1
        @test injectivity_radius(OrthogonalMatrices(1; parameter = :field)) == 0.0
    end
end
