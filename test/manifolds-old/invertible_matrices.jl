using LinearAlgebra, Manifolds, ManifoldsBase, Test, Random

@testset "Invertible matrices" begin
    @test_throws DomainError InvertibleMatrices(-1)
    M = InvertibleMatrices(3, ℝ)
    A = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    B = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Mc = InvertibleMatrices(3, ℂ)
    Ac = [1.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Bc = [0.0im 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @testset "Real invertible matrices" begin
        @test repr(M) == "InvertibleMatrices(3, ℝ)"
        M2 = InvertibleMatrices(3, ℝ; parameter = :field)
        @test repr(M2) == "InvertibleMatrices(3, ℝ; parameter=:field)"
        @test check_point(M, A) == nothing
        @test_throws DomainError is_point(M, B; error = :error)
        @test_throws ManifoldDomainError is_point(M, Ac; error = :error)
        @test_throws ManifoldDomainError is_vector(M, A, Ac; error = :error)
        @test is_vector(M, A, A)
        @test is_flat(M)
        @test typeof(get_embedding(M)) ===
            Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3, 3}}}
        @test typeof(get_embedding(M2)) === Euclidean{ℝ, Tuple{Int64, Int64}}
        @test embed(M, A) === A
        @test embed(M, A, A) === A
        @test manifold_dimension(M) == 9
        @test Weingarten(M, A, A, A) == zero(A)
        @test riemann_tensor(M, A, A, B, A) == zero(A)
        @test sectional_curvature(M, A, A, B) === 0.0
        @test sectional_curvature(Mc, Ac, Ac, im .* Ac) === 0.0
        @test sectional_curvature_max(M) === 0.0
        @test sectional_curvature_min(M) === 0.0
        @test Weingarten!(M, similar(A), A, A, A) == zero(A)

        @test is_point(M, rand(M))
        @test is_point(M, rand(Random.MersenneTwister(), M))
        @test is_vector(M, A, rand(M; vector_at = A))

        @test get_coordinates(M, A, A, DefaultOrthonormalBasis()) == vec(A)
        c = similar(vec(A))
        get_coordinates!(M, c, A, A, DefaultOrthonormalBasis())
        @test isapprox(c, vec(A))

        @test get_vector(M, A, vec(A), DefaultOrthonormalBasis()) == A
        D = similar(A)
        get_vector!(M, D, A, vec(A), DefaultOrthonormalBasis())
        @test isapprox(D, A)

        @test get_coordinates(M, A, A, DefaultOrthogonalBasis()) == vec(A)
        @test get_vector(M, A, vec(A), DefaultOrthogonalBasis()) == A
    end
    @testset "tangent projection" begin
        Z = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        @test project(M, A, Z) == Z
        @test project(Mc, Ac, im * Z) == im * Z
        Y = zero(Z)
        @test project!(M, Y, A, Z) === Y
        @test Y == Z
    end
    @testset "Complex invertible matrices" begin
        @test repr(Mc) == "InvertibleMatrices(3, ℂ)"
        Mc2 = InvertibleMatrices(3, ℂ; parameter = :field)
        @test repr(Mc2) == "InvertibleMatrices(3, ℂ; parameter=:field)"
        @test manifold_dimension(Mc) == 2 * 3^2
        @test check_point(Mc, Ac) == nothing
        @test_throws DomainError is_point(Mc, Bc; error = :error)
        @test_throws DomainError is_point(Mc, B; error = :error)
        @test is_point(Mc, A; error = :error)
        cr = [real(vec(Ac)); imag(vec(Ac))]
        @test get_coordinates(Mc, Ac, Ac, DefaultOrthonormalBasis()) == cr
        @test get_vector(Mc, Ac, cr, DefaultOrthonormalBasis()) == Ac
        for Bs in [DefaultOrthogonalBasis(), DefaultOrthonormalBasis(ℂ)]
            @test get_vector(Mc, Ac, get_coordinates(Mc, Ac, Ac, Bs), Bs) == Ac
        end
    end
end
