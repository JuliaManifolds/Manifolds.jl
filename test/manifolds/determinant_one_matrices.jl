using Manifolds, Random, Test

@testset "Invertible Matrices of Determinant one" begin
    @testset "Real case" begin
        M = DeterminantOneMatrices(2)
        # is det 1 and inv
        @test is_point(M, [1.0 0.0; 0.0 1.0]; error=:error)
        # det 1 but for example not Rotation
        @test is_point(M, [10.0 0.0; 0.0 0.1]; error=:error)
        # Not invertible
        @test_throws DomainError is_point(M, [10.0 0.0; 0.0 0.0]; error=:error)
        # Det(2)
        @test_throws DomainError is_point(M, [2.0 0.0; 0.0 1.0]; error=:error)
        Random.seed!(42)
        p = rand(M)
        @test is_point(M, p)
        X = rand(M; vector_at=p)
        @test is_vector(M, p, X)
        # not trace 0
        Xf = [1.0 1.1; 1.2 0.0]
        @test_throws DomainError is_vector(M, p, Xf; error=:error)
        @test get_embedding(M) == Euclidean(2, 2)
        @test manifold_dimension(M) == 3
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
        Xf = [1.0+2.0im 1.1+0.1im; 1.2 0.0]
        @test_throws DomainError is_vector(M, p, Xf; error=:error)
        @test repr(M) == "DeterminantOneMatrices(2, ℂ)"
    end

    @testset "Field parameter" begin
        M = DeterminantOneMatrices(2; parameter=:field)
        @test repr(M) == "DeterminantOneMatrices(2, ℝ; parameter=:field)"
    end
end
