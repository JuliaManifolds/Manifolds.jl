include("../utils.jl")

@testset "Generalized Stiefel" begin
    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B)
        x = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        @testset "Basics" begin
            @test repr(M) ==
                  "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 3
            @test base_manifold(M) === M
            @test_throws DomainError is_point(M, [1.0, 0.0, 0.0, 0.0], true)
            @test_throws DomainError is_point(M, 1im * [1.0 0.0; 0.0 1.0; 0.0 0.0], true)
            @test !is_vector(M, x, [0.0, 0.0, 1.0, 0.0])
            @test_throws DomainError is_vector(M, x, [0.0, 0.0, 1.0, 0.0], true)
            @test_throws DomainError is_vector(M, x, 1 * im * zero_vector(M, x), true)
        end
        @testset "Embedding and Projection" begin
            y = similar(x)
            z = embed(M, x)
            @test z == x
            embed!(M, y, x)
            @test y == z
            a = [1.0 0.0; 0.0 2.0; 0.0 0.0]
            @test !is_point(M, a)
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
        X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
        Y = [0.0 0.0; 0.0 0.0; -1.0 1.0]
        @test inner(M, x, X, Y) == 0
        y = retract(M, x, X)
        z = retract(M, x, Y)
        @test is_point(M, y)
        @test is_point(M, z)
        a = project(M, x + X)
        b = retract(M, x, X)
        c = retract(M, x, X, ProjectionRetraction())
        d = retract(M, x, X, PolarRetraction())
        @test a == b
        @test c == d
        @test b == c
        e = similar(a)
        retract!(M, e, x, X)
        @test e == a
        @test vector_transport_to(M, x, X, y, ProjectionTransport()) == project(M, y, X)
        @testset "Type $T" for T in types
            pts = convert.(T, [x, y, z])
            @test !is_point(M, 2 * x)
            @test_throws DomainError !is_point(M, 2 * x, true)
            @test !is_vector(M, x, y)
            @test_throws DomainError is_vector(M, x, y, true)
            test_manifold(
                M,
                pts,
                test_exp_log=false,
                default_inverse_retraction_method=nothing,
                default_retraction_method=ProjectionRetraction(),
                test_injectivity_radius=false,
                test_is_tangent=true,
                test_project_tangent=true,
                test_default_vector_transport=false,
                test_forward_diff=false,
                test_reverse_diff=false,
                projection_atol_multiplier=15.0,
                retraction_atol_multiplier=10.0,
                is_tangent_atol_multiplier=4 * 10.0^2,
                # investigate why this is so large on 1.6 dev
                exp_log_atol_multiplier=10.0^3 * (VERSION >= v"1.6-DEV" ? 10.0^8 : 1.0),
                retraction_methods=[PolarRetraction(), ProjectionRetraction()],
                mid_point12=nothing,
                test_inplace=true,
            )
        end
    end

    @testset "Complex" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B, ℂ)
        @testset "Basics" begin
            @test repr(M) ==
                  "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℂ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 8
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
        M = GeneralizedStiefel(3, 2, B, ℍ)
        @test repr(M) ==
              "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℍ)"
        @testset "Basics" begin
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 18
        end
    end
end
