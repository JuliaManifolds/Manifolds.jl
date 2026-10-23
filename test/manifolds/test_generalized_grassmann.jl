using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The generalized Grassmann manifold" begin
    @test_throws DomainError GeneralizedGrassmann(2, 3)

    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B)
        p1 = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
        X2 = [0.0 0.0; 0.0 0.0; 0.5 0.0]
        p2 = exp(M, p1, X)
        p3 = exp(M, p1, [0.0 0.0; 0.0 0.0; 0.3 -0.2])
        W = [0.1 0.2; -0.3 0.4; 0.5 -0.6]
        # Invalids
        pf1 = 2 * p1
        pf2 = [1.0, 0.0, 0.0, 0.0]
        pf3 = 1im * [1.0 0.0; 0.0 1.0; 0.0 0.0]
        Xf1 = [0.0 0.0; 1.0 0.0; 0.0 2.0] # p1' * B * Xf1 ≠ 0
        Xf2 = [0.0, 0.0, 1.0, 0.0]
        Xf3 = 1im * zero_vector(M, p1)

        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [[1.0 0.0; 0.0 2.0; 0.0 0.0]],
                :EmbeddedVectors => [W],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                    default_vector_transport_method, distance, embed, exp, geodesic,
                    get_embedding, injectivity_radius, inner, inverse_retract, is_flat,
                    is_point, is_vector, log, manifold_dimension, mid_point, norm, project,
                    rand, repr, representation_size, retract, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, zero_vector,
                ],
                :InvalidPoints => [pf1, pf2, pf3],
                :InvalidVectors => [Xf1, Xf2, Xf3],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), PolarInverseRetraction(), missing,
                ],
                :Points => [p1, p2, p3],
                :RetractionMethods => [
                    ExponentialRetraction(), PolarRetraction(), ProjectionRetraction(),
                ],
                :SecondVector => X2,
                :Vectors => [X, project(M, p2, X2), project(M, p3, X)],
                :VectorTransportMethods => [ProjectionTransport()],
                :Seed => 42,
                :TestRandScaling => true,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-13,
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ProjectionTransport(),
                distance => sqrt(2),
                (embed, :Point) => p1,
                (embed, :Vector) => X,
                get_embedding => GeneralizedStiefel(3, 2, B),
                injectivity_radius => π / 2,
                (injectivity_radius, ExponentialRetraction()) => π / 2,
                (injectivity_radius, PolarRetraction()) => π / 2,
                (injectivity_radius, ProjectionRetraction()) => π / 2,
                inner => -0.5,
                is_flat => false,
                log => X,
                manifold_dimension => 2,
                norm => sqrt(2),
                (project, :Point) => p1,
                (project, :Vector) => W - p1 * (p1' * B * W),
                (retract, ProjectionRetraction()) => project(M, p1 + X),
                (retract, PolarRetraction()) => project(M, p1 + X),
                repr => "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)",
                representation_size => (3, 2),
                (vector_transport_to, ProjectionTransport()) => project(M, p2, X),
                :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
                :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )

        @testset "Basics" begin
            @test M == GeneralizedGrassmann(3, 2, copy(B))
            @test hash(M) == hash(GeneralizedGrassmann(3, 2, copy(B)))
            @test M != GeneralizedGrassmann(3, 2, B, ℂ)
            Bs = [1.0 -0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
            @test !isequal(M, GeneralizedGrassmann(3, 2, Bs))
            @test base_manifold(M) === M
            @test injectivity_radius(M, p1, ExponentialRetraction()) == π / 2
            @test mean(M, [p1, p1, p1]) == p1
            @test !is_vector(M, p1, p2)
            @test_throws ManifoldDomainError is_vector(M, p1, p2; error = :error)
        end

        @testset "Embedding and Projection" begin
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_point(M, a)
            @test project(M, a) == p1
            b = similar(a)
            project!(M, b, a)
            @test b == p1
            q = project(M, [1.0 0.0; 0.0 1.0; 1.0 1.0])
            @test project(M, q, W) ≈ W - q * (q' * B * W)
            @test is_vector(M, q, project(M, q, W))
            W3 = rand(MersenneTwister(42), M; vector_at = q)
            @test W3 ≈ W3 - q * (q' * B * W3)
        end

        @testset "retractions and vector transport" begin
            Y = [0.0 0.0; 0.0 0.0; 1.0 -0.5]
            @test retract(M, p1, Y) == exp(M, p1, Y)
            @test retract(M, p1, Y, ProjectionRetraction()) == project(M, p1 + Y)
            @test retract(M, p1, Y, ProjectionRetraction()) == retract(M, p1, Y, PolarRetraction())
            @test vector_transport_to(M, p1, Y, p2) == project(M, p2, Y)
            Zg = inverse_retract(M, p1, p2, PolarInverseRetraction())
            @test norm(p1' * M.B * Zg) < 1.0e-14
            @test distance(M, retract(M, p1, Zg, PolarRetraction()), p2) < 1.0e-14
        end

        @testset "gradient and metric conversion" begin
            L = cholesky(B).L
            Y = [0.0 0.0; 0.0 0.0; 1.0 -1.0]
            @test change_metric(M, EuclideanMetric(), p1, Y) == L \ Y
            @test change_representer(M, EuclideanMetric(), p1, Y) == B \ Y
        end
    end

    @testset "geodesic speed" begin
        B = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B)
        p = project(M, [1.0 0.0; 0.0 1.0; 0.0 0.0])
        X = [0.1 0.2; -0.3 0.4; 0.5 -0.6]
        X = X - p * (p' * B * X)
        for t in [0.25, 0.5, 1.0]
            @test distance(M, p, exp(M, p, t * X)) ≈ t * norm(M, p, X)
        end
        @test isapprox(M, p, log(M, p, exp(M, p, X)), X)
    end

    @testset "distance and logarithm" begin
        B = [2.0 1.0 0.5; 1.0 3.0 0.7; 0.5 0.7 1.5]
        M = GeneralizedGrassmann(3, 2, B)
        p = project(M, [1.0 0.0; 0.0 1.0; 0.0 0.0])
        q = project(M, [0.0 1.0; 1.0 0.0; 1.0 1.0])
        @test distance(M, p, q) ≈ distance(M, q, p)
        # cosines of the principal angles in the inner product of B
        @test distance(M, p, q) ≈ norm(acos.(min.(svdvals(p' * B * q), 1)))
        X = log(M, p, q)
        @test norm(M, p, X) ≈ distance(M, p, q)
        @test isapprox(M, exp(M, p, X), q)
        # a right angle in the inner product of B
        M3 = GeneralizedGrassmann(4, 2, [1.0 0 0 0; 0 1 0 0; 0 0 4 0; 0 0 0 1])
        p3 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        q3 = [0.0 0.0; 0.0 1.0; 0.5 0.0; 0.0 0.0]
        @test distance(M3, p3, q3) ≈ π / 2
    end

    @testset "change_metric for a non-diagonal B" begin
        B = [2.0 1.0 0.5; 1.0 3.0 0.7; 0.5 0.7 1.5]
        M = GeneralizedGrassmann(3, 2, B)
        p = project(M, [1.0 0.0; 0.0 1.0; 0.0 0.0])
        X = [0.1 0.2; -0.3 0.4; 0.5 -0.6]
        X = X - p * (p' * B * X)
        Z = [0.3 -0.1; 0.2 0.5; -0.4 0.1]
        Z = Z - p * (p' * B * Z)
        Y = change_metric(M, EuclideanMetric(), p, X)
        W = change_metric(M, EuclideanMetric(), p, Z)
        @test inner(M, p, Y, Y) ≈ dot(X, X)
        @test inner(M, p, Y, W) ≈ dot(X, Z)
        @test Y - p * (p' * B * Y) ≈ Y
        @test change_metric(M, EuclideanMetric(), p, Y) ≈ B \ X - p * (p' * X)
    end

    @testset "Complex" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B, ℂ)
        p1 = [1im 0.0; 0.0 0.5im; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; 0.0 1.0]
        X2 = [0.0 0.0; 0.0 0.0; 1.0im 0.0]
        p2 = exp(M, p1, X)
        p3 = exp(M, p1, [0.0 0.0; 0.0 0.0; 0.3 -0.2im])
        F = ComplexF64[1.0 2.0im; 3.0 4.0; 5.0im 6.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [F],
                :Functions => [
                    copy, copyto!, distance, embed, exp, get_embedding, injectivity_radius,
                    inner, is_flat, is_point, is_vector, log, manifold_dimension, norm,
                    project, rand, repr, representation_size, zero_vector,
                ],
                :InvalidPoints => [2 * p1, [1.0, 0.0, 0.0, 0.0]],
                :InvalidVectors => [[0.0, 0.0, 1.0, 0.0]],
                :Points => [p1, p2, p3],
                :SecondVector => X2,
                :Vectors => [X, project(M, p2, X2), project(M, p3, X)],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-13,
                distance => 1.0,
                get_embedding => GeneralizedStiefel(3, 2, B, ℂ),
                injectivity_radius => π / 2,
                is_flat => false,
                log => X,
                manifold_dimension => 4,
                norm => 1.0,
                (number_eltype, rand) => ComplexF64,
                repr => "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℂ)",
                representation_size => (3, 2),
                :IsPointErrors => [ManifoldDomainError, ManifoldDomainError],
                :IsVectorErrors => [ManifoldDomainError],
            ),
        )

        @testset "rand and project with the identity as B" begin
            N = GeneralizedGrassmann(3, 2, Matrix{ComplexF64}(I, 3, 3), ℂ)
            p = rand(MersenneTwister(42), N)
            @test eltype(p) == ComplexF64
            @test is_point(N, p)
            @test is_vector(N, p, rand(MersenneTwister(43), N; vector_at = p))
            @test is_point(N, project(N, F))
        end
    end

    @testset "Quaternion" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B, ℍ)
        Manifolds.Test.test_manifold(
            M,
            Dict(:Functions => [is_flat, manifold_dimension, repr, representation_size]),
            # Expectations
            Dict(
                is_flat => false,
                manifold_dimension => 8,
                repr => "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℍ)",
                representation_size => (3, 2),
            ),
        )
    end

    @testset "small distance tests" begin
        n, k = 3, 2
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        @testset for fT in (Float32, Float64), T in (fT, Complex{fT})
            𝔽 = T <: Complex ? ℂ : ℝ
            M = GeneralizedGrassmann(n, k, B, 𝔽)
            rT = real(T)
            atol = rtol = sqrt(eps(rT))
            @testset for t in (zero(rT), eps(rT)^(1 // 4) / 8, eps(rT)^(1 // 4))
                p = project(M, randn(T, representation_size(M)))
                X = project(M, p, randn(T, representation_size(M)))
                X ./= norm(M, p, X)
                project!(M, X, p, X)
                @test distance(M, p, exp(M, p, t * X)) ≈ t atol = atol rtol = rtol
            end
        end
    end

    @testset "field parameter" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B; parameter = :field)
        @test repr(M) ==
            "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ; parameter=:field)"
        @test typeof(get_embedding(M)) ===
            GeneralizedStiefel{ℝ, Tuple{Int64, Int64}, Matrix{Float64}}
    end
end
