include("utils.jl")

@testset "SymmetricMatrices" begin
    M=SymmetricMatrices(3,Real)
    A = [1 2 3; 4 5 6; 7 8 9]
    A_sym = [1 2 3; 2 5 -1; 3 -1 9]
    B_sym = [1 2 3; 2 5 1; 3 1 -1]
    X = zeros(3,3)
    @testset "Real Symmetric Matrices Basics" begin
        @test representation_size(M) == (3,3)
        @test check_manifold_point(M,B_sym)==nothing
        @test_throws DomainError is_manifold_point(M,A,true)
        @test check_tangent_vector(M,B_sym,B_sym)==nothing
        @test_throws DomainError is_tangent_vector(M,B_sym,A,true)
        @test manifold_dimension(M) == 6
    end
    types = [ Matrix{Float32},
            Matrix{Float64},
            MMatrix{3,3,Float32},
            MMatrix{3,3,Float64},
            SizedMatrix{3,3,Float32},
            SizedMatrix{3,3,Float64},
        ]
    for T in types
        pts = [convert(T,A_sym),convert(T,B_sym),convert(T,X)]
        @testset "Type $T" begin
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector),
                          test_project_tangent = true,
                          test_musical_isomorphisms = true,
                          test_vector_transport = true
                          )
            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end # testset type $T
    end # for 
end # test SymmetricMatrices
