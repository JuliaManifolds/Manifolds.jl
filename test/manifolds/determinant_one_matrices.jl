using Manifolds, Random, Test

@testset "Invertible Matrices of Determinant one" begin
    @testset "Real case" begin
        M = DeterminantOneMatrices(2)
        # is det 1 and inv
        @test is_point(M, [1.0 0.0; 0.0 1.0], true)
        # det 1 but for example not Rotation
        @test is_point(M, [10.0 0.0; 0.0 0.1], true)
        # Not invertible
        @test_throws DomainError is_point(M, [10.0 0.0; 0.0 0.0], true)
        # Det(2)
        @test_throws DomainError is_point(M, [2.0 0.0; 0.0 1.0], true)
        Random.seed!(42)
        p = rand(M)
        @test is_point(M, p)
        X = rand(M; vector_at=p)
        @test is_vector(M, p, X)
        @test repr(M) == "DeterminantOneMatrices(2, ℝ)"
    end
    @testset "Complex case" begin
        M = DeterminantOneMatrices(2, ℂ)
        # is det 1 and inv
        @test is_point(M, [1.0im 0.0; 0.0 -1.0im], true)
        # det 1 but for example not Rotation
        @test is_point(M, [10.0im 0.0; 0.0 -0.1im], true)
        # Not invertible
        @test_throws DomainError is_point(M, [10.0 0.0; 0.0 0.0], true)
        # Det(2)
        @test_throws DomainError is_point(M, [2.0 0.0; 0.0 1.0], true)
        Random.seed!(42)
        p = rand(M)
        @test is_point(M, p)
        X = rand(M; vector_at=p)
        @test is_vector(M, p, X)
        @test repr(M) == "DeterminantOneMatrices(2, ℂ)"
    end
end
