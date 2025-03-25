using Manifolds, Random, Test, LinearAlgebra

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

    @testset "Field parameter" begin
        M = DeterminantOneMatrices(2; parameter=:field)
        @test repr(M) == "DeterminantOneMatrices(2, ℝ; parameter=:field)"
    end
    @testset "rng test" begin
        M = DeterminantOneMatrices(2)
        pX = zeros(2, 2)
        Manifolds._ensure_nonzero_rng_determinant!(
            Random.default_rng(),
            get_embedding(M),
            pX,
        )
        @test abs(det(pX)) > 1e-8
    end
end
