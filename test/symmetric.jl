include("utils.jl")

@testset "SymmetricMatrices" begin
    M = SymmetricMatrices(3, ℝ)
    A = [1 2 3; 4 5 6; 7 8 9]
    A_sym = [1 2 3; 2 5 -1; 3 -1 9]
    A_sym2 = [1 2 3; 2 5 -1; 3 -1 9]
    B_sym = [1 2 3; 2 5 1; 3 1 -1]
    M_complex = SymmetricMatrices(3, ℂ)
    @test repr(M_complex) == "SymmetricMatrices(3, ℂ)"
    @test Manifolds.allocation_promotion_function(M_complex, get_vector, ()) === complex
    C = [1 1 -im; 1 2 -im; im im -1]
    D = [1 0; 0 1]
    X = zeros(3, 3)
    @testset "Real Symmetric Matrices Basics" begin
        @test repr(M) == "SymmetricMatrices(3, ℝ)"
        @test representation_size(M) == (3, 3)
        @test base_manifold(M) === M
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3,3},ℝ}
        @test check_manifold_point(M, B_sym) === nothing
        @test_throws DomainError is_manifold_point(M, A, true)
        @test_throws DomainError is_manifold_point(M, C, true)
        @test_throws DomainError is_manifold_point(M, D, true)
        @test check_tangent_vector(M, B_sym, B_sym) === nothing
        @test_throws DomainError is_tangent_vector(M, B_sym, A, true)
        @test_throws DomainError is_tangent_vector(M, A, B_sym, true)
        @test_throws DomainError is_tangent_vector(M, B_sym, D, true)
        @test_throws DomainError is_tangent_vector(
            M,
            B_sym,
            1 * im * zero_tangent_vector(M, B_sym),
            true,
        )
        @test manifold_dimension(M) == 6
        @test manifold_dimension(M_complex) == 9
        @test A_sym2 == project!(M, A_sym, A_sym)
        @test A_sym2 == project(M, A_sym, A_sym)
        A_sym3 = similar(A_sym)
        embed!(M, A_sym3, A_sym)
        A_sym4 = embed(M, A_sym)
        @test A_sym3 == A_sym
        @test A_sym4 == A_sym
    end
    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{3,3,Float64,9})

    bases = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        pts = [convert(T, A_sym), convert(T, B_sym), convert(T, X)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                basis_types_vecs = (
                    DiagonalizingOrthonormalBasis(log(M, pts[1], pts[2])),
                    bases...,
                ),
                basis_types_to_from = bases,
            )
            test_manifold(
                M_complex,
                pts,
                test_injectivity_radius = false,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                basis_types_vecs = (DefaultOrthonormalBasis(ℂ),),
                basis_types_to_from = (DefaultOrthonormalBasis(ℂ),),
            )
            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end # testset type $T
    end # for
end # test SymmetricMatrices
