include("../header.jl")

@testset "Multinomial symmetric positive definite matrices" begin
    @testset "Basics" begin
        M = MultinomialSymmetricPositiveDefinite(3)
        Mf = MultinomialSymmetricPositiveDefinite(3; parameter = :field)
        @test repr(M) == "MultinomialSymmetricPositiveDefinite(3)"
        @test repr(Mf) == "MultinomialSymmetricPositiveDefinite(3; parameter=:field)"
        @test get_embedding(M) == MultinomialSymmetric(3)
        @test get_embedding(Mf) == MultinomialSymmetric(3; parameter = :field)
        @test manifold_dimension(M) == manifold_dimension(MultinomialSymmetric(3)) == 3
        @test manifold_dimension(Mf) == 3
        @test !is_flat(M)
        @test is_flat(MultinomialSymmetricPositiveDefinite(2))
        #
        # Checks
        # (a) Points
        p = [0.6 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.6]
        @test is_point(M, p; error = :error)
        # Symmetric but does not sum to 1
        pf1 = zeros(3, 3)
        @test_throws ManifoldDomainError is_point(M, pf1; error = :error)
        #  in theory this is not spd since it has an EV 0 but numerically it is
        pf2 = [0.3 0.4 0.3; 0.4 0.2 0.4; 0.3 0.4 0.3]
        # Multinomial but not symmetric
        pf3 = [0.2 0.3 0.5; 0.4 0.2 0.4; 0.4 0.5 0.1]
        @test_throws ManifoldDomainError is_point(M, pf3; error = :error)
        # (b) Tangent vectors
        X = [1.0 -0.5 -0.5; -0.5 1.0 -0.5; -0.5 -0.5 1.0]
        @test is_vector(M, p, X; error = :error)
        Xf1 = ones(3, 3) # Symmetric but does not sum to zero
        @test_throws ManifoldDomainError is_vector(M, p, Xf1; error = :error)
        #sums to zero but not symmetric
        Xf2 = [-0.5 0.3 0.2; 0.2 -0.5 0.3; 0.3 0.2 -0.5]
        @test_throws ManifoldDomainError is_vector(M, p, Xf2; error = :error)
    end

    @testset "Random" begin
        q = zeros(3, 3)
        M = MultinomialSymmetricPositiveDefinite(3)
        @test is_point(M, rand!(MersenneTwister(), M, q); error = :error, atol = 5.0e-14)
    end

    @testset "Fisher metric, tangent projection and retraction" begin
        M = MultinomialSymmetricPositiveDefinite(3)
        N = MultinomialSymmetric(3)
        p = [0.5 0.3 0.2; 0.3 0.4 0.3; 0.2 0.3 0.5]
        Y = [0.1 0.2 -0.3; 0.2 0.1 0.05; -0.3 0.05 0.1]
        G = [1.0 2.0 3.0; 0.5 -1.0 2.0; 1.5 0.0 -2.0]
        X = project(M, p, Y)
        @test X ≈ project(N, p, Y)
        @test inner(M, p, X, X) ≈ inner(N, p, X, X)
        @test norm(M, p, X) ≈ norm(N, p, X)
        @test riemannian_gradient(M, p, G) ≈ riemannian_gradient(N, p, G)
        q = retract(M, p, 0.1 * X, ProjectionRetraction())
        @test q ≈ retract(N, p, 0.1 * X, ProjectionRetraction())
        @test ManifoldsBase.retract_fused(M, p, X, 0.1, ProjectionRetraction()) ≈ q
        @test is_point(N, q)
        @test isposdef(Symmetric(q))
    end
end
