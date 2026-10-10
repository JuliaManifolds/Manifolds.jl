include("../header.jl")

@testset "Isochoric matrices" begin
    M = SPDFixedDeterminant(2, 1.0)
    @test repr(M) == "SPDFixedDeterminant(2, 1.0)"
    p = [1.0 0.0; 0.0 1.0]
    @test is_point(M, p)
    # Determinant is 4
    @test !is_point(M, 2.0 .* p)
    @test_throws DomainError is_point(M, 2.0 .* p; error = :error)
    #
    X = [0.0 0.1; 0.1 0.0]
    @test is_vector(M, p, X)
    Y = [1.0 0.1; 0.1 1.0]
    @test !is_vector(M, p, Y)
    @test_throws DomainError is_vector(M, p, Y; error = :error)

    @test project(M, 2.0 .* p) == p
    @test project(M, p, Y) == X

    @test embed(M, p) == p
    @test embed(M, p, X) == X
    q = zero(p)
    @test embed!(M, q, p) == p
    @test p == q
    Y = zero(X)
    @test embed!(M, Y, p, X) == X
    @test Y == X

    @test manifold_dimension(M) == 2

    q = exp(M, p, X)
    @test det(q) ≈ 1
    @test distance(M, q, exp(get_embedding(M), p, X)) ≈ 0 atol = 6.0e-16
    @test norm(M, p, log(M, p, q) - X) ≈ 0 atol = 3.0e-16
    @test norm(M, p, log(get_embedding(M), p, q) - X) ≈ 0 atol = 3.0e-16

    @testset "tangent vectors away from the identity" begin
        M3 = SPDFixedDeterminant(3, 1.0)
        p3 = [2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.5]
        # trace zero, but the geodesic in this direction changes the determinant
        X3 = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]
        @test !is_vector(M3, p3, X3)
        @test project(M3, p3, X3) ≈ [4.0 0.0 0.0; 0.0 -2.5 0.0; 0.0 0.0 0.25] ./ 3
        q3 = project(M3, [4.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 2.0])
        Z3 = [1.0 2.0 3.0; 2.0 -1.0 0.5; 3.0 0.5 4.0]
        @test is_vector(M3, q3, project(M3, q3, 100 .* Z3))
        @test is_vector(M3, p3, log(M3, p3, q3))
    end

    @testset "field parameter" begin
        M = SPDFixedDeterminant(2, 1.0; parameter = :field)
        @test repr(M) == "SPDFixedDeterminant(2, 1.0; parameter=:field)"
        @test get_embedding(M) == SymmetricPositiveDefinite(2; parameter = :field)
    end
    @testset "random points and tangent vectors" begin
        M = SPDFixedDeterminant(2, 1.0)
        p = rand(MersenneTwister(42), M)
        @test is_point(M, p)
        @test is_vector(M, p, rand(MersenneTwister(44), M; vector_at = p))
    end
end
