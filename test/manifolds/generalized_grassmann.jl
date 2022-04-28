include("../utils.jl")

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
            @test_throws DomainError is_point(M, [1.0, 0.0, 0.0, 0.0], true)
            @test_throws DomainError is_point(M, 1im * [1.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test_throws DomainError is_point(M, 2 * p, true)
            @test !is_vector(M, p, [0.0, 0.0, 1.0, 0.0])
            @test_throws DomainError is_vector(M, p, [0.0, 0.0, 1.0, 0.0], true)
            @test_throws DomainError is_vector(M, p, 1 * im * zero_vector(M, p), true)
            @test_throws DomainError is_vector(M, p, X, true)
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, p) == π / 2
            @test injectivity_radius(M, p, ExponentialRetraction()) == π / 2
            @test mean(M, [p, p, p]) == p
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
        TEST_STATIC_SIZED && push!(types, MMatrix{3,2,Float64,6})
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
        @testset "Type $T" for T in types
            pts = convert.(T, [p, q, r])
            @test !is_point(M, 2 * p)
            @test_throws DomainError !is_point(M, 2 * r, true)
            @test !is_vector(M, p, q)
            @test_throws DomainError is_vector(M, p, q, true)
            test_manifold(
                M,
                pts,
                test_exp_log=false,
                default_inverse_retraction_method=LogarithmicInverseRetraction(),
                default_retraction_method=ExponentialRetraction(),
                test_injectivity_radius=false,
                test_is_tangent=true,
                test_project_tangent=true,
                test_default_vector_transport=false,
                test_forward_diff=false,
                test_reverse_diff=false,
                projection_atol_multiplier=15.0,
                retraction_atol_multiplier=10.0,
                is_tangent_atol_multiplier=4 * 10.0^2,
                retraction_methods=[PolarRetraction(), ProjectionRetraction()],
                mid_point12=nothing,
                test_inplace=true,
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
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            x = [1.0 0.0; 0.0 0.5; 0.0 0.0]

            x = [1im 0.0; 0.0 0.5im; 0.0 0.0]
            @test is_point(M, x)
            @test !is_point(M, 2 * x)
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
        end
    end
end
