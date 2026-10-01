include("../header.jl")

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
        @test is_flat(M)
        @test typeof(get_embedding(M)) === Euclidean{ℝ, TypeParameter{Tuple{3, 3}}}
        @test check_point(M, B_sym) === nothing
        @test_throws DomainError is_point(M, A; error = :error)
        @test_throws ManifoldDomainError is_point(M, C; error = :error)
        @test_throws ManifoldDomainError is_point(M, D; error = :error) #embedding changes type
        @test check_vector(M, B_sym, B_sym) === nothing
        @test_throws DomainError is_vector(M, B_sym, A; error = :error)
        @test_throws ManifoldDomainError is_vector(M, A, B_sym; error = :error)
        @test_throws ManifoldDomainError is_vector(M, B_sym, D; error = :error)
        @test_throws ManifoldDomainError is_vector(
            M,
            B_sym,
            1 * im * zero_vector(M, B_sym);
            error = :error,
        )
        @test manifold_dimension(M) == 6
        @test manifold_dimension(M_complex) == 9
        @test A_sym2 == project!(M, A_sym, A_sym)
        @test A_sym2 == project(M, A_sym, A_sym)
        @test project(M, A) == [1 3 5; 3 5 7; 5 7 9] # the point projection symmetrizes
        @test project(M, B_sym, A) == [1 3 5; 3 5 7; 5 7 9] # and so does the tangent one
        A_sym3 = similar(A_sym)
        embed!(M, A_sym3, A_sym)
        A_sym4 = embed(M, A_sym)
        @test A_sym3 == A_sym
        @test A_sym4 == A_sym
        # the projection onto the tangent space is the Hermitian part
        C_sym = ComplexF64[1 2 + 3im 0; 2 - 3im 4 0; 0 0 5]
        @test is_vector(M_complex, C_sym, C_sym)
        @test project(M_complex, C_sym, C_sym) == C_sym
    end
    types = [Matrix{Float64}]

    bases = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        pts = [convert(T, A_sym), convert(T, B_sym), convert(T, X)]
        @testset "Type $T" begin
            Manifolds.test_manifold(
                M,
                pts,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                basis_types_vecs = (
                    DiagonalizingOrthonormalBasis(log(M, pts[1], pts[2])),
                    bases...,
                ),
                basis_types_to_from = bases,
                is_tangent_atol_multiplier = 1,
                test_inplace = true,
            )
            pts_complex = [
                convert(Matrix{ComplexF64}, A_sym) + [0 im 2im; -im 0 -im; -2im im 0],
                convert(Matrix{ComplexF64}, B_sym) + [0 -2im im; 2im 0 3im; -im -3im 0],
                convert(Matrix{ComplexF64}, X),
            ]
            Manifolds.test_manifold(
                M_complex,
                pts_complex,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                basis_types_vecs = (DefaultOrthonormalBasis(ℝ),),
                basis_types_to_from = (DefaultOrthonormalBasis(ℝ),),
                is_tangent_atol_multiplier = 1,
                test_inplace = true,
            )
            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end # testset type $T
    end # for
    @testset "complex coordinates" begin
        Y = ComplexF64[1 2 + 3im 0; 2 - 3im 4 0; 0 0 5]
        c = get_coordinates(M_complex, Y, Y, DefaultOrthonormalBasis())
        Z = get_vector(M_complex, Y, c, DefaultOrthonormalBasis())
        @test is_vector(M_complex, Y, Z)
        @test isapprox(Z, Y)
    end
    @testset "field parameter" begin
        M = SymmetricMatrices(3, ℝ; parameter = :field)
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(M) == "SymmetricMatrices(3, ℝ; parameter=:field)"
    end
end # test SymmetricMatrices
