include("../header.jl")

@testset "generalized Grassmann geodesic speed" begin
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

@testset "generalized Grassmann distance and logarithm" begin
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

@testset "generalized Grassmann change_metric" begin
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

@testset "Generalized Grassmann" begin
    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B)
        p = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        X = zeros(3, 2)
        X[1, :] .= 1.0
        @testset "Basics" begin
            @test repr(M) ==
                "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 2
            @test base_manifold(M) === M
            @test !is_flat(M)
            @test_throws ManifoldDomainError is_point(M, [1.0, 0.0, 0.0, 0.0]; error = :error)
            @test_throws ManifoldDomainError is_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0];
                error = :error,
            )
            @test_throws ManifoldDomainError is_point(M, 2 * p; error = :error)
            @test !is_vector(M, p, [0.0, 0.0, 1.0, 0.0])
            @test_throws ManifoldDomainError is_vector(
                M,
                p,
                [0.0, 0.0, 1.0, 0.0];
                error = :error,
            )
            @test_throws ManifoldDomainError is_vector(
                M,
                p,
                1 * im * zero_vector(M, p);
                error = :error,
            )
            @test_throws ManifoldDomainError is_vector(M, p, X; error = :error)
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, p) == π / 2
            @test injectivity_radius(M, p, ExponentialRetraction()) == π / 2
            @test mean(M, [p, p, p]) == p
            Random.seed!(42)
            @test is_point(M, rand(M))
            @test is_vector(M, p, rand(M; vector_at = p))
        end
        @testset "Embedding and Projection" begin
            q = similar(p)
            p2 = embed(M, p)
            @test p2 == p
            embed!(M, q, p)
            @test q == p2
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_point(M, a)
            b = similar(a)
            c = project(M, a)
            @test c == p
            project!(M, b, a)
            @test b == p
            X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Y = similar(X)
            Z = embed(M, p, X)
            embed!(M, Y, p, X)
            @test Y == X
            @test Z == X
            p3 = project(M, [1.0 0.0; 0.0 1.0; 1.0 1.0])
            W = [0.1 0.2; -0.3 0.4; 0.5 -0.6]
            @test project(M, p3, W) ≈ W - p3 * (p3' * B * W)
            @test is_vector(M, p3, project(M, p3, W))
            W3 = rand(MersenneTwister(42), M; vector_at = p3)
            @test W3 ≈ W3 - p3 * (p3' * B * W3)
        end
        @testset "gradient and metric conversion" begin
            L = cholesky(B).L
            X = [0.0 0.0; 0.0 0.0; 1.0 -1.0]
            Y = change_metric(M, EuclideanMetric(), p, X)
            @test Y == L \ X
            Z = change_representer(M, EuclideanMetric(), p, X)
            @test Z == B \ X
        end
        types = [Matrix{Float64}]
        X = [0.0 0.0; 1.0 0.0; 0.0 2.0]
        Y = [0.0 0.0; -1.0 0.0; 0.0 2.0]
        @test inner(M, p, X, Y) == 0
        q = retract(M, p, X)
        r = retract(M, p, Y)
        @test is_point(M, q)
        @test is_point(M, r)
        @test retract(M, p, X) == exp(M, p, X)

        a = project(M, p + X)
        c = retract(M, p, X, ProjectionRetraction())
        d = retract(M, p, X, PolarRetraction())
        @test a == c
        @test c == d
        e = similar(a)
        retract!(M, e, p, X)
        @test e == exp(M, p, X)
        @test vector_transport_to(M, p, X, q, ProjectionTransport()) == project(M, q, X)
        @test default_vector_transport_method(M) === ProjectionTransport()
        @test vector_transport_to(M, p, X, q) == project(M, q, X)
        @testset "Type $T" for T in types
            pts = convert.(T, [p, q, r])
            @test !is_point(M, 2 * p)
            @test_throws ManifoldDomainError is_point(M, 2 * r; error = :error)
            @test !is_vector(M, p, q)
            @test_throws ManifoldDomainError is_vector(M, p, q; error = :error)
            Manifolds.test_manifold(
                M,
                pts,
                test_exp_log = false,
                default_inverse_retraction_method = LogarithmicInverseRetraction(),
                default_retraction_method = ExponentialRetraction(),
                test_injectivity_radius = false,
                test_is_tangent = true,
                test_project_tangent = true,
                test_default_vector_transport = false,
                projection_atol_multiplier = 15.0,
                retraction_atol_multiplier = 10.0,
                is_point_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 4 * 10.0^2,
                retraction_methods = [PolarRetraction(), ProjectionRetraction()],
                mid_point12 = nothing,
                test_inplace = true,
            )
        end
    end

    @testset "Complex" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B, ℂ)
        @testset "Basics" begin
            @test repr(M) ==
                "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℂ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 4
            @test !is_flat(M)
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            x = [1im 0.0; 0.0 0.5im; 0.0 0.0]
            @test is_point(M, x)
            @test !is_point(M, 2 * x)
        end
        @testset "rand and project" begin
            F = ComplexF64[1.0 2.0im; 3.0 4.0; 5.0im 6.0]
            @testset for N in (M, GeneralizedGrassmann(3, 2, Matrix{ComplexF64}(I, 3, 3), ℂ))
                p = rand(MersenneTwister(42), N)
                @test eltype(p) == ComplexF64
                @test is_point(N, p)
                @test is_vector(N, p, rand(MersenneTwister(43), N; vector_at = p))
                @test is_point(N, project(N, F))
            end
        end
    end

    @testset "Quaternion" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B, ℍ)
        @test repr(M) ==
            "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℍ)"
        @testset "Basics" begin
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 8
            @test !is_flat(M)
        end
    end

    @testset "small distance tests" begin
        n, k = 3, 2
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        @testset for fT in (Float32, Float64), T in (fT, Complex{fT})
            𝔽 = T isa Complex ? ℂ : ℝ
            M = GeneralizedGrassmann(n, k, B, 𝔽)
            rT = real(T)
            atol = eps(rT)^(1 // 4)
            rtol = eps(rT)
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
