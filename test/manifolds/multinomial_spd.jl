include("../header.jl")

@testset "Multinomial symmetric positive definite matrices" begin
    @testset "Basics" begin
        M = MultinomialSymmetricPositiveDefinite(3)
        Mf = MultinomialSymmetricPositiveDefinite(3; parameter=:field)
        @test repr(M) == "MultinomialSymmetricPositiveDefinite(3)"
        @test repr(Mf) == "MultinomialSymmetricPositiveDefinite(3; parameter=:field)"
        @test get_embedding(M) == MultinomialMatrices(3, 3)
        @test get_embedding(Mf) == MultinomialMatrices(3, 3; parameter=:field)
        #
        # Checks
        # (a) Points
        p = [0.6 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.6]
        @test is_point(M, p; error=:error)
        # Symmetric but does not sum to 1
        pf1 = zeros(3, 3)
        @test_throws ManifoldDomainError is_point(M, pf1; error=:error)
        #  in theory this is not spd since it has an EV 0 but numerically it is
        pf2 = [0.3 0.4 0.3; 0.4 0.2 0.4; 0.3 0.4 0.3]
        # Multinomial but not symmetric
        pf3 = [0.2 0.3 0.5; 0.4 0.2 0.4; 0.4 0.5 0.1]
        @test_throws ManifoldDomainError is_point(M, pf3; error=:error)
        # (b) Tangent vectors
        X = [1.0 -0.5 -0.5; -0.5 1.0 -0.5; -0.5 -0.5 1.0]
        @test is_vector(M, p, X; error=:error)
        Xf1 = ones(3, 3) # Symmetric but does not sum to zero
        @test_throws ManifoldDomainError is_vector(M, p, Xf1; error=:error)
        #sums to zero but not symmetric
        Xf2 = [-0.5 0.3 0.2; 0.2 -0.5 0.3; 0.3 0.2 -0.5]
        @test_throws ManifoldDomainError is_vector(M, p, Xf2; error=:error)
    end

    @testset "Random" begin
        q = zeros(3, 3)
        M = MultinomialSymmetricPositiveDefinite(3)
        @test is_point(M, rand!(MersenneTwister(), M, q); atol=4e-15)
    end
end
