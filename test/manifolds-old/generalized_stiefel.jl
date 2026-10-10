include("../header.jl")

@testset "Generalized Stiefel" begin
    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B)
        p = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        X = zeros(3, 2)
        X[1, :] .= 1
        @testset "Basics" begin
            @test repr(M) ==
                "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)"
            @test representation_size(M) == (3, 2)
            @test manifold_dimension(M) == 3
            @test base_manifold(M) === M
            @test !is_flat(M)
            @test_throws DomainError is_point(M, [1.0, 0.0, 0.0, 0.0]; error = :error)
            @test_throws ManifoldDomainError is_point(
                M,
                1im * [1.0 0.0; 0.0 1.0; 0.0 0.0];
                error = :error,
            )
            @test_throws DomainError is_point(M, 2 * p; error = :error)
            @test !is_vector(M, p, [0.0, 0.0, 1.0, 0.0])
            @test_throws DomainError is_vector(M, p, [0.0, 0.0, 1.0, 0.0]; error = :error)
            @test_throws ManifoldDomainError is_vector(
                M,
                p,
                1 * im * zero_vector(M, p);
                error = :error,
            )
            @test_throws DomainError is_vector(M, p, X; error = :error)
            @test default_retraction_method(M) == ProjectionRetraction()
            @test is_point(M, rand(M))
            @test is_vector(M, p, rand(M; vector_at = p))
            M1 = GeneralizedStiefel(3, 1, B)
            p1 = project(M1, reshape([1.0, 1.0, 1.0], 3, 1))
            @test is_vector(M1, p1, project(M1, p1, reshape([0.3, -0.2, 0.5], 3, 1)))
        end
        @testset "Embedding and Projection" begin
            @test get_embedding(GeneralizedStiefel(3, 2)) == Euclidean(3, 2)
            y = similar(p)
            z = embed(M, p)
            @test z == p
            embed!(M, y, p)
            @test y == z
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

        types = [Matrix{Float64}]
        X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
        Y = [0.0 0.0; 0.0 0.0; -1.0 1.0]
        @test inner(M, p, X, Y) == 0
        y = retract(M, p, X)
        z = retract(M, p, Y)
        @test is_point(M, y)
        @test is_point(M, z)
        a = project(M, p + X)
        b = retract(M, p, X)
        c = retract(M, p, X, ProjectionRetraction())
        d = retract(M, p, X, PolarRetraction())
        @test a == b
        @test b == c
        # the polar factor d of p + X with respect to B: d'B(p + X) is symmetric positive definite
        @test is_point(M, d)
        @test d' * B * (p + X) ≈ (d' * B * (p + X))'
        @test isposdef(Symmetric(d' * B * (p + X)))
        e = similar(a)
        retract!(M, e, p, X)
        @test e == a
        @test vector_transport_to(M, p, X, y, ProjectionTransport()) == project(M, y, X)
        @testset "Polar, QR and Cayley retractions and the inverses" begin
            M2 = GeneralizedStiefel(4, 2, Diagonal([1.0, 2.0, 3.0, 4.0]))
            p2 = project(M2, [1.0 0.0; 1.0 1.0; 1.0 2.0; 0.0 3.0])
            X2 = project(M2, p2, [0.1 0.2; -0.3 0.4; 0.5 0.6; 0.7 -0.8])
            q2 = retract(M2, p2, X2, PolarRetraction())
            @test is_point(M2, q2)
            @test isapprox(M2, p2, inverse_retract(M2, p2, q2, PolarInverseRetraction()), X2)
            r2 = retract(M2, p2, X2, QRRetraction())
            @test is_point(M2, r2)
            # r2 = (p2 + X2)R^{-1}, so r2'B(p2 + X2) = R is upper triangular with positive diagonal
            R2 = r2' * M2.B * (p2 + X2)
            @test isapprox(tril(R2, -1), zeros(2, 2); atol = 1.0e-14)
            @test all(diag(R2) .> 0)
            @test isapprox(M2, p2, inverse_retract(M2, p2, r2, QRInverseRetraction()), X2)
            c2 = retract(M2, p2, X2, CayleyRetraction())
            @test is_point(M2, c2)
            @test retract(M2, p2, zero(X2), CayleyRetraction()) == p2
            @test retract(M2, p2, X2 / 2, CayleyRetraction()) ≈
                ManifoldsBase.retract_fused(M2, p2, X2, 0.5, CayleyRetraction())
        end
        @testset "Type $T" for T in types
            pts = convert.(T, [p, y, z])
            @test !is_point(M, 2 * p)
            @test_throws DomainError is_point(M, 2 * z; error = :error)
            @test !is_vector(M, p, y)
            @test_throws DomainError is_vector(M, p, y; error = :error)
            Manifolds.test_manifold(
                M,
                pts,
                test_exp_log = false,
                default_inverse_retraction_method = nothing,
                default_retraction_method = ProjectionRetraction(),
                test_injectivity_radius = false,
                test_is_tangent = true,
                test_project_tangent = true,
                test_default_vector_transport = false,
                projection_atol_multiplier = 15.0,
                retraction_atol_multiplier = 10.0,
                is_tangent_atol_multiplier = 4 * 10.0^2,
                # investigate why this is so large on 1.6 dev
                exp_log_atol_multiplier = 10.0^3 * (VERSION >= v"1.6-DEV" ? 10.0^8 : 1.0),
                retraction_methods = [PolarRetraction(), ProjectionRetraction()],
                mid_point12 = nothing,
                test_inplace = true,
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
            @test !is_flat(M)
            p = rand(MersenneTwister(42), M)
            @test eltype(p) == ComplexF64
            @test is_point(M, p)
            @test is_vector(M, p, rand(MersenneTwister(43), M; vector_at = p))
            @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
            @test !is_vector(M, [1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0, 1.0, 0.0])
            x = [1im 0.0; 0.0 0.5im; 0.0 0.0]
            @test is_point(M, x)
            @test !is_point(M, 2 * x)
            Xc = project(M, x, [0.1 0.2im; -0.3 0.4; 0.5im 0.6])
            @test is_vector(M, x, Xc)
            @test !is_vector(M, x, [0.0 1.0; 1.0 0.0; 0.0 0.0])
            @test !is_vector(M, x, 1.0e-9 * [0.0 1.0; 1.0 0.0; 0.0 0.0])
            F = ComplexF64[1.0 2.0im; 3.0 4.0; 5.0im 6.0]
            @test is_point(M, project(M, F))
            MI = GeneralizedStiefel(3, 2, Matrix{ComplexF64}(I, 3, 3), ℂ)
            @test is_point(MI, project(MI, F))
            @test is_point(M, retract(M, x, Xc, PolarRetraction()))
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
            @test !is_flat(M)
        end
    end

    @testset "field parameter" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B; parameter = :field)
        @test repr(M) ==
            "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ; parameter=:field)"
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int64, Int64}}
    end
end
