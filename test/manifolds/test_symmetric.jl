using LinearAlgebra, Manifolds, ManifoldsBase, Test

@testset "SymmetricMatrices" begin
    @test_throws DomainError SymmetricMatrices(-1)
    M = SymmetricMatrices(3, ℝ)
    M_complex = SymmetricMatrices(3, ℂ)
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    A_sym = [1.0 2.0 3.0; 2.0 5.0 -1.0; 3.0 -1.0 9.0]
    B_sym = [1.0 2.0 3.0; 2.0 5.0 1.0; 3.0 1.0 -1.0]
    C = [1 1 -im; 1 2 -im; im im -1]
    D = [1.0 0.0; 0.0 1.0]

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, flat, geodesic, get_basis,
        get_coordinates, get_embedding, get_vector, get_vectors, injectivity_radius, inner,
        inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
        parallel_transport_direction, parallel_transport_to, project, rand, repr,
        representation_size, retract, sectional_curvature, sharp, shortest_geodesic,
        vector_transport_direction, vector_transport_to, Weingarten, zero_vector,
    ]
    common_expectations = Dict(
        :atol => 1.0e-12,
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ParallelTransport(),
        (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
        injectivity_radius => Inf,
        (injectivity_radius, ExponentialRetraction()) => Inf,
        is_flat => true,
        representation_size => (3, 3),
        sectional_curvature => 0.0,
        sectional_curvature_max => 0.0,
        sectional_curvature_min => 0.0,
        Weingarten => zeros(3, 3),
    )

    @testset "Real symmetric matrices" begin
        X = B_sym - A_sym
        Y = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [collect(1.0:6.0)],
                :Covectors => [flat(M, A_sym, X)],
                :EmbeddedPoints => [A],
                :EmbeddedVectors => [A],
                :Functions => functions,
                :InvalidPoints => Any[A, C, D], # Any avoids promotion to a common type
                :InvalidVectors => Any[A, D, 1im * zero_vector(M, A_sym)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                # skew-symmetric matrices are normal to the symmetric ones
                :NormalVectors => [[0.0 1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0]],
                :Points => [A_sym, B_sym, zeros(3, 3)],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, Y, X],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    distance => sqrt(108),
                    (embed, :Point) => A_sym,
                    (embed, :Vector) => X,
                    exp => B_sym,
                    (get_coordinates, DefaultOrthonormalBasis()) => [0.0, 0.0, 0.0, 0.0, 2 * sqrt(2), -10.0],
                    get_embedding => Euclidean(3, 3),
                    inner => 4.0,
                    log => X,
                    manifold_dimension => 6,
                    mid_point => (A_sym + B_sym) / 2,
                    norm => sqrt(108),
                    parallel_transport_direction => X,
                    parallel_transport_to => X,
                    # the projections symmetrize
                    (project, :Point) => [1.0 3.0 5.0; 3.0 5.0 7.0; 5.0 7.0 9.0],
                    (project, :Vector) => [1.0 3.0 5.0; 3.0 5.0 7.0; 5.0 7.0 9.0],
                    repr => "SymmetricMatrices(3, ℝ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => X,
                    :IsPointErrors => [DomainError, ManifoldDomainError, ManifoldDomainError],
                    :IsVectorErrors => [DomainError, ManifoldDomainError, ManifoldDomainError],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )

        @testset "bases only available through get_basis" begin
            for B in [DiagonalizingOrthonormalBasis(X), ProjectedOrthonormalBasis(:svd)]
                b = get_basis(M, A_sym, B)
                @test b isa CachedBasis
                V = get_vectors(M, A_sym, b)
                @test length(V) == 6
                @test all(is_vector(M, A_sym, v) for v in V)
                @test [inner(M, A_sym, v, w) for v in V, w in V] ≈ I(6)
                @test get_vector(M, A_sym, get_coordinates(M, A_sym, Y, b), b) ≈ Y
            end
        end
        @test isapprox(-A_sym, exp(M, A_sym, log(M, A_sym, -A_sym)))
    end

    @testset "Complex Hermitian matrices" begin
        p1 = A_sym + [0 im 2im; -im 0 -im; -2im im 0]
        p2 = B_sym + [0 -2im im; 2im 0 3im; -im -3im 0]
        X = p2 - p1
        Y = ComplexF64[0 im 0; -im 0 0; 0 0 1]
        Manifolds.Test.test_manifold(
            M_complex,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [collect(1.0:9.0)],
                :Covectors => [flat(M_complex, p1, X)],
                :EmbeddedPoints => [complex(A)],
                :EmbeddedVectors => [complex(A)],
                :Functions => functions,
                :InvalidPoints => Any[complex(A), D],
                :InvalidVectors => Any[complex(A), D],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :NormalVectors => [ComplexF64[0 1 0; -1 0 0; 0 0 0]],
                :Points => [p1, p2, zeros(ComplexF64, 3, 3)],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, Y, X],
                :VectorTransportMethods => [
                    ParallelTransport(), SchildsLadderTransport(), PoleLadderTransport(),
                ],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    distance => norm(X),
                    (embed, :Point) => p1,
                    (embed, :Vector) => X,
                    exp => p2,
                    get_embedding => Euclidean(3, 3; field = ℂ),
                    inner => dot(X, Y),
                    log => X,
                    manifold_dimension => 9,
                    mid_point => (p1 + p2) / 2,
                    norm => norm(X),
                    (number_eltype, rand) => ComplexF64,
                    parallel_transport_direction => X,
                    parallel_transport_to => X,
                    # the projections take the Hermitian part
                    (project, :Point) => ComplexF64[1 3 5; 3 5 7; 5 7 9],
                    (project, :Vector) => ComplexF64[1 3 5; 3 5 7; 5 7 9],
                    repr => "SymmetricMatrices(3, ℂ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => X,
                    :IsPointErrors => [DomainError, ManifoldDomainError],
                    :IsVectorErrors => [DomainError, ManifoldDomainError],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
    end

    @testset "Basics" begin
        @test Manifolds.allocation_promotion_function(M_complex, get_vector, ()) === complex
        @test base_manifold(M) === M
        @test riemann_tensor(M, A_sym, A_sym, B_sym, B_sym) == zeros(3, 3)
        @test sectional_curvature(M, A_sym, A_sym, B_sym) === 0.0
        @test sectional_curvature(M_complex, C, C, B_sym) === 0.0
        @test typeof(get_embedding(M)) === Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3, 3}}}
        @test check_point(M, B_sym) === nothing
        @test check_vector(M, B_sym, B_sym) === nothing
        @test Weingarten!(M, similar(B_sym), B_sym, B_sym, B_sym) == zero(B_sym)
        A_sym2 = copy(A_sym)
        @test A_sym2 == project!(M, A_sym2, A_sym2)
        @test A_sym == project(M, A_sym, A_sym)
        # the projection onto the tangent space is the Hermitian part
        C_sym = ComplexF64[1 2 + 3im 0; 2 - 3im 4 0; 0 0 5]
        @test is_vector(M_complex, C_sym, C_sym)
        @test project(M_complex, C_sym, C_sym) == C_sym
    end

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
end
