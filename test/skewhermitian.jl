include("utils.jl")

@testset "SkewSymmetricMatrices" begin
    @test SkewSymmetricMatrices(3) === SkewHermitianMatrices(3)
    @test SkewSymmetricMatrices(3, ℂ) === SkewHermitianMatrices(3, ℂ)
end

@testset "SkewHermitianMatrices" begin
    M = SkewHermitianMatrices(3, ℝ)
    A = [1 2 3; 4 5 6; 7 8 9]
    A_skewsym = [0 -2 -3; 2 0 1; 3 -1 0]
    A_skewsym2 = [0 -2 -3; 2 0 1; 3 -1 0]
    B_skewsym = [0 -2 -3; 2 0 -1; 3 1 0]
    M_complex = SkewHermitianMatrices(3, ℂ)
    A_skewsym_complex = [1.0im -2.0 -3.0; 2.0 0.0 1.0; 3.0 -1.0 0.0]
    B_skewsym_complex = [2.0im -2.0-2.0im -3.0; 2.0-2.0im 0.0 -1.0; 3.0 1.0 0.0]
    @test repr(M_complex) == "SkewHermitianMatrices(3, ℂ)"
    C = [0 -1 im; 1 0 -im; im -im 0]
    D = [1 0; 0 1]
    X = zeros(3, 3)
    @testset "Real Skew-Symmetric Matrices Basics" begin
        @test repr(M) == "SkewSymmetricMatrices(3)"
        @test representation_size(M) == (3, 3)
        @test base_manifold(M) === M
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3,3},ℝ}
        @test check_point(M, B_skewsym) === nothing
        @test_throws DomainError is_manifold_point(M, A, true)
        @test_throws DomainError is_manifold_point(M, C, true)
        @test_throws DomainError is_manifold_point(M, D, true)
        @test check_tangent_vector(M, B_skewsym, B_skewsym) === nothing
        @test_throws DomainError is_tangent_vector(M, B_skewsym, A, true)
        @test_throws DomainError is_tangent_vector(M, A, B_skewsym, true)
        @test_throws DomainError is_tangent_vector(M, B_skewsym, D, true)
        @test_throws DomainError is_tangent_vector(
            M,
            B_skewsym,
            1 * im * zero_vector(M, B_skewsym),
            true,
        )
        @test manifold_dimension(M) == 3
        @test manifold_dimension(M_complex) == 9
        @test A_skewsym2 == project!(M, A_skewsym, A_skewsym)
        @test A_skewsym2 == project(M, A_skewsym, A_skewsym)
        A_sym3 = similar(A_skewsym)
        embed!(M, A_sym3, A_skewsym)
        A_sym4 = embed(M, A_skewsym)
        @test A_sym3 == A_skewsym
        @test A_sym4 == A_skewsym
    end
    types = [Matrix{Float64}]
    TEST_FLOAT32 && push!(types, Matrix{Float32})
    TEST_STATIC_SIZED && push!(types, MMatrix{3,3,Float64,9})
    bases = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        pts = [convert(T, A_skewsym), convert(T, B_skewsym), convert(T, X)]
        @testset "Type $T" begin
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_reverse_diff=isa(T, Vector),
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                basis_types_vecs=(
                    DiagonalizingOrthonormalBasis(log(M, pts[1], pts[2])),
                    bases...,
                ),
                basis_types_to_from=bases,
                is_tangent_atol_multiplier=1,
            )
        end
    end
    complex_types = [Matrix{ComplexF64}]
    TEST_FLOAT32 && push!(complex_types, Matrix{ComplexF32})
    TEST_STATIC_SIZED && push!(complex_types, MMatrix{3,3,ComplexF64,9})
    for T in complex_types
        pts_complex =
            [convert(T, A_skewsym_complex), convert(T, B_skewsym_complex), convert(T, X)]
        @testset "Type $T" begin
            test_manifold(
                M_complex,
                pts_complex,
                test_injectivity_radius=false,
                test_reverse_diff=isa(T, Vector),
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                basis_types_vecs=(DefaultOrthonormalBasis(ℂ),),
                basis_types_to_from=(DefaultOrthonormalBasis(ℂ),),
                is_tangent_atol_multiplier=1,
            )
            @test isapprox(
                -pts_complex[1],
                exp(M, pts_complex[1], log(M, pts_complex[1], -pts_complex[1])),
            )
        end # testset type $T
    end # for
end # test SymmetricMatrices
