include("utils.jl")

@testset "SymmetricMatrices" begin
    M=SymmetricMatrices(3,ℝ)
    A = [1 2 3; 4 5 6; 7 8 9]
    A_sym = [1 2 3; 2 5 -1; 3 -1 9]
    A_sym2 = [1 2 3; 2 5 -1; 3 -1 9]
    B_sym = [1 2 3; 2 5 1; 3 1 -1]
    M_complex = SymmetricMatrices(3,ℂ)
    @test repr(M_complex) == "SymmetricMatrices(3, ℂ)"
    C = [1+im 1 im; 1 2 -im; im -im -1-im]
    D = [1 0; 0 1];
    X = zeros(3,3)
    @testset "Real Symmetric Matrices Basics" begin
        @test repr(M) == "SymmetricMatrices(3, ℝ)"
        @test representation_size(M) == (3,3)
        @test check_manifold_point(M,B_sym)===nothing
        @test_throws DomainError is_manifold_point(M,A,true)
        @test_throws DomainError is_manifold_point(M,C,true)
        @test_throws DomainError is_manifold_point(M,D,true)
        @test_throws DomainError is_manifold_point(M_complex, [:a :b :c; :b :d :e; :c :e :f],true)
        @test check_tangent_vector(M,B_sym,B_sym)===nothing
        @test_throws DomainError is_tangent_vector(M,B_sym,A,true)
        @test_throws DomainError is_tangent_vector(M,A,B_sym,true)
        @test_throws DomainError is_tangent_vector(M,B_sym,D,true)
        @test_throws DomainError is_tangent_vector(M,B_sym, 1*im * zero_tangent_vector(M,B_sym),true)
        @test_throws DomainError is_tangent_vector(M_complex, B_sym, [:a :b :c; :b :d :e; :c :e :f],true)
        @test manifold_dimension(M) == 6
        @test manifold_dimension(M_complex) == 12
        @test A_sym2 == project_point!(M,A_sym)
        @test A_sym2 == project_tangent(M,A_sym,A_sym)
    end
    types = [
        Matrix{Float64},
        MMatrix{3,3,Float64},
        Matrix{Float32},
    ]
    bases = (ArbitraryOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        pts = [convert(T,A_sym),convert(T,B_sym),convert(T,X)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                basis_types_vecs = bases,
                basis_types_to_from = bases
            )
            test_manifold(
                M_complex,
                pts,
                test_injectivity_radius = false,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                basis_types_vecs = (ArbitraryOrthonormalBasis(),),
                basis_types_to_from = (ArbitraryOrthonormalBasis(),)
            )
            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end # testset type $T
    end # for
end # test SymmetricMatrices
