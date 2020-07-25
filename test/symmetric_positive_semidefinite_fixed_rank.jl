include("utils.jl")

@testset "Symmetric Positive Semidefinite Matrices of Fixed Rank" begin
    @testset "Real Matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4,2)
        @test repr(M) == "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℝ)"
        @test manifold_dimension(M) == 7
        q = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        @test is_manifold_point(M,q)
        Y = [1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
        @test_throws DomainError is_manifold_point(M,Y, true)
        @test is_tangent_vector(M,q,Y)
        q2 = [2.0 1.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
        q3 = [0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 0.0]
    end
    @testset "Complex Matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4,2,ℂ)
        @test repr(M) == "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℂ)"
        @test manifold_dimension(M) == 12

    end
end
