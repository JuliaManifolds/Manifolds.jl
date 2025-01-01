using LinearAlgebra, Manifolds, ManifoldsBase, Test, Random

@testset "Invertible matrices" begin
    M = InvertibleMatrices(3, ℝ)
    A = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    B = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Mc = InvertibleMatrices(3, ℂ)
    Ac = [1.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Bc = [0.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @testset "Real invertible matrices" begin
        @test repr(M) == "InvertibleMatrices(3, ℝ)"
        M2 = InvertibleMatrices(3, ℝ; parameter=:field)
        @test repr(M2) == "InvertibleMatrices(3, ℝ; parameter=:field)"
        @test check_point(M, A) == nothing
        @test_throws DomainError is_point(M, B; error=:error)
        @test_throws ManifoldDomainError is_point(M, Ac; error=:error)
        @test_throws ManifoldDomainError is_vector(M, A, Ac; error=:error)
        @test is_vector(M, A, A)
        @test is_flat(M)
        @test typeof(get_embedding(M)) ===
              Euclidean{ManifoldsBase.TypeParameter{Tuple{3,3}},ℝ}
        @test typeof(get_embedding(M2)) === Euclidean{Tuple{Int64,Int64},ℝ}
        @test embed(M, A) === A
        @test embed(M, A, A) === A
        @test manifold_dimension(M) == 9
        @test Weingarten(M, A, A, A) == zero(A)

        @test is_point(M, rand(M))
        @test is_point(M, rand(Random.MersenneTwister(), M))
        @test is_vector(M, A, rand(M; vector_at=A))

        @test get_coordinates(M, A, A, DefaultOrthonormalBasis()) == vec(A)
        @test get_vector(M, A, vec(A), DefaultOrthonormalBasis()) == A
    end
    @testset "Complex invertible matrices" begin
        @test repr(Mc) == "InvertibleMatrices(3, ℂ)"
        Mc2 = InvertibleMatrices(3, ℂ; parameter=:field)
        @test repr(Mc2) == "InvertibleMatrices(3, ℂ; parameter=:field)"
        @test manifold_dimension(Mc) == 2 * 3^2
        @test check_point(Mc, Ac) == nothing
        @test_throws DomainError is_point(Mc, Bc; error=:error)
        @test_throws DomainError is_point(Mc, B; error=:error)
        @test is_point(Mc, A; error=:error)
    end
end
