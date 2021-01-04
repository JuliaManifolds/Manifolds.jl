include("utils.jl")

@testset "Generalized Grassmann" begin
    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedGrassmann(3, 2, B)
        x = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        @testset "Basics" begin
            @test repr(M) ==
                  "GeneralizedGrassmann(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 2
            @test base_manifold(M) === M
            @test_throws DomainError is_manifold_point(M, [1.0, 0.0, 0.0, 0.0], true)
            @test_throws DomainError is_manifold_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0],
                true,
            )
            @test !is_tangent_vector(M, x, [0.0, 0.0, 1.0, 0.0])
            @test_throws DomainError is_tangent_vector(M, x, [0.0, 0.0, 1.0, 0.0], true)
            @test_throws DomainError is_tangent_vector(
                M,
                x,
                1 * im * zero_tangent_vector(M, x),
                true,
            )
            @test injectivity_radius(M) == π / 2
            @test injectivity_radius(M, ExponentialRetraction()) == π / 2
            @test injectivity_radius(M, x) == π / 2
            @test injectivity_radius(M, x, ExponentialRetraction()) == π / 2
            @test mean(M, [x, x, x]) == x
        end
        @testset "Embedding and Projection" begin
            y = similar(x)
            z = embed(M, x)
            @test z == x
            embed!(M, y, x)
            @test y == z
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_manifold_point(M, a)
            b = similar(a)
            c = project(M, a)
            @test c == x
            project!(M, b, a)
            @test b == x
            X = [0.0 0.0; 0.0 0.0; -1.0 1.0]
            Y = similar(X)
            Z = embed(M, x, X)
            embed!(M, Y, x, X)
            @test Y == X
            @test Z == X
        end

        types = [Matrix{Float64}]
        TEST_STATIC_SIZED && push!(types, MMatrix{3,2,Float64,6})
        X = [0.0 0.0; 1.0 0.0; 0.0 2.0]
        Y = [0.0 0.0; -1.0 0.0; 0.0 2.0]
        @test inner(M, x, X, Y) == 0
        y = retract(M, x, X)
        z = retract(M, x, Y)
        @test is_manifold_point(M, y)
        @test is_manifold_point(M, z)
        @test retract(M, x, X) == exp(M, x, X)

        a = project(M, x + X)
        c = retract(M, x, X, ProjectionRetraction())
        d = retract(M, x, X, PolarRetraction())
        @test a == c
        @test c == d
        e = similar(a)
        retract!(M, e, x, X)
        @test e == exp(M, x, X)
        @test vector_transport_to(M, x, X, y, ProjectionTransport()) == project(M, y, X)
        @testset "Type $T" for T in types
            pts = convert.(T, [x, y, z])
            @test !is_manifold_point(M, 2 * x)
            @test_throws DomainError !is_manifold_point(M, 2 * x, true)
            @test !is_tangent_vector(M, x, y)
            @test_throws DomainError is_tangent_vector(M, x, y, true)
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
            @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_tangent_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            x = [1.0 0.0; 0.0 0.5; 0.0 0.0]

            x = [1im 0.0; 0.0 0.5im; 0.0 0.0]
            @test is_manifold_point(M, x)
            @test !is_manifold_point(M, 2 * x)
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
