include("../utils.jl")

@testset "Multinomial symmetric positive definite matrices" begin
    @testset "Basics" begin
        M = MultinomialSymmetricPositiveDefinite(3)
        Mf = MultinomialSymmetricPositiveDefinite(3; parameter=:field)
        @test repr(M) == "MultinomialSymmetricPositiveDefinite(3)"
        @test repr(Mf) == "MultinomialSymmetricPositiveDefinite(3; parameter=:field)"
        @test get_embedding(M) == MultinomialMatrices(3, 3)
        @test get_embedding(Mf) == MultinomialMatrices(3, 3; parameter=:field)
    end

    @testset "Random" begin
        q = zeros(3, 3)
        M = MultinomialSymmetricPositiveDefinite(3)
        @test is_point(M, rand!(MersenneTwister(), M, q))
    end
end
