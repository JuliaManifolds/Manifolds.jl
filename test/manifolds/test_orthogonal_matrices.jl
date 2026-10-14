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
                manifold_dimension,
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
        @test manifold_volume(OrthogonalMatrices(3)) ≈ 32 * π^2 * sqrt(2)
        @test manifold_volume(OrthogonalMatrices(4)) ≈ 16 * (2 * π)^4
        @test manifold_volume(OrthogonalMatrices(5)) ≈ 256 * (2 * π)^6 / 6
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

    @testset "Inverse retractions on both connected components" begin
        rng = MersenneTwister(42)
        refl(n) = Diagonal([-1; ones(n - 1)])
        methods = [
            (QRRetraction(), QRInverseRetraction()),
            (PolarRetraction(), PolarInverseRetraction()),
        ]
        for (rm, irm) in methods, n in [2, 3, 5]
            On = OrthogonalMatrices(n)
            SOn = Rotations(n)
            for s in [1, -1]
                p = s == 1 ? rand(rng, SOn) : rand(rng, SOn) * refl(n)
                X = 2 .* rand(rng, On; vector_at = p)
                q = retract(On, p, X, rm)
                @test det(q) ≈ det(p)
                Y = inverse_retract(On, p, q, irm)
                @test is_vector(On, p, Y)
                @test isapprox(On, p, X, Y; atol = 1.0e-12)
                @test isapprox(On, q, retract(On, p, Y, rm))
                Y2 = zero_vector(On, p)
                inverse_retract!(On, Y2, p, q, irm)
                @test Y2 ≈ Y
                # points from different connected components
                @test_throws DomainError inverse_retract(On, p, q * refl(n), irm)
                @test_throws DomainError inverse_retract(On, p * refl(n), q, irm)
            end
            # on Rotations the result agrees with the one on OrthogonalMatrices
            p = rand(rng, SOn)
            q = retract(SOn, p, rand(rng, SOn; vector_at = p), rm)
            @test inverse_retract(SOn, p, q, irm) ≈
                inverse_retract(On, p, q, irm)
        end
        @test_throws DomainError inverse_retract(M, p1, p2, QRInverseRetraction())
        @test_throws DomainError inverse_retract(M, p1, p2, PolarInverseRetraction())
    end

    @testset "Random tangent vectors scale with σ" begin
        X = rand(MersenneTwister(42), M; vector_at = p1)
        Y = rand(MersenneTwister(42), M; vector_at = p1, σ = 2.0)
        @test is_vector(M, p1, Y)
        @test Y == 2 .* X
    end
end
