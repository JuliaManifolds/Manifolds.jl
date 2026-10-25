using LinearAlgebra, Manifolds, ManifoldsBase, Test

@testset "SkewHermitianMatrices" begin
    @test_throws DomainError SkewSymmetricMatrices(-1)
    @test SkewSymmetricMatrices(3) === SkewHermitianMatrices(3)
    M = SkewHermitianMatrices(3, ℝ)
    M_complex = SkewHermitianMatrices(3, ℂ)
    A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    A_skewsym = [0.0 -2.0 -3.0; 2.0 0.0 1.0; 3.0 -1.0 0.0]
    B_skewsym = [0.0 -2.0 -3.0; 2.0 0.0 -1.0; 3.0 1.0 0.0]
    C = [0 -1 im; 1 0 -im; im -im 0]
    D = [1.0 0.0; 0.0 1.0]

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, flat, geodesic, get_basis,
        get_coordinates, get_embedding, get_vector, get_vectors, injectivity_radius, inner,
        inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension, mid_point, norm,
        parallel_transport_direction, parallel_transport_to, project, rand, repr,
        representation_size, retract, sectional_curvature, sharp, shortest_geodesic,
        vector_transport_direction, vector_transport_to, zero_vector,
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
    )

    @testset "Real skew-symmetric matrices" begin
        X = B_skewsym - A_skewsym
        Y = [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 -1.0 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [DefaultOrthonormalBasis()],
                :Coordinates => [[1.0, 2.0, 3.0]],
                :Covectors => [flat(M, A_skewsym, X)],
                :EmbeddedPoints => [A],
                :EmbeddedVectors => [A],
                :Functions => [functions; Weingarten],
                :InvalidPoints => Any[A, C, D], # Any avoids promotion to a common type
                :InvalidVectors => Any[A, D, 1im * zero_vector(M, A_skewsym)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                # symmetric matrices are normal to the skew-symmetric ones
                :NormalVectors => [[1.0 2.0 0.0; 2.0 0.0 0.0; 0.0 0.0 3.0]],
                :Points => [A_skewsym, B_skewsym, zeros(3, 3)],
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
                    distance => sqrt(8),
                    (embed, :Point) => A_skewsym,
                    (embed, :Vector) => X,
                    exp => B_skewsym,
                    (get_coordinates, DefaultOrthonormalBasis()) => [0.0, 0.0, -2 * sqrt(2)],
                    get_embedding => Euclidean(3, 3),
                    (get_vector, [1.0, 2.0, 3.0], DefaultOrthonormalBasis()) =>
                        [0.0 1.0 2.0; -1.0 0.0 3.0; -2.0 -3.0 0.0] ./ sqrt(2),
                    inner => -4.0,
                    log => X,
                    manifold_dimension => 3,
                    mid_point => (A_skewsym + B_skewsym) / 2,
                    norm => sqrt(8),
                    parallel_transport_direction => X,
                    parallel_transport_to => X,
                    # the projections antisymmetrize
                    (project, :Point) => [0.0 -1.0 -2.0; 1.0 0.0 -1.0; 2.0 1.0 0.0],
                    (project, :Vector) => [0.0 -1.0 -2.0; 1.0 0.0 -1.0; 2.0 1.0 0.0],
                    repr => "SkewSymmetricMatrices(3)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => X,
                    Weingarten => zeros(3, 3),
                    :IsPointErrors => [DomainError, ManifoldDomainError, ManifoldDomainError],
                    :IsVectorErrors => [DomainError, ManifoldDomainError, ManifoldDomainError],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )

        @testset "bases only available through get_basis" begin
            for B in [DiagonalizingOrthonormalBasis(X), ProjectedOrthonormalBasis(:svd)]
                b = get_basis(M, A_skewsym, B)
                @test b isa CachedBasis
                V = get_vectors(M, A_skewsym, b)
                @test length(V) == 3
                @test all(is_vector(M, A_skewsym, v) for v in V)
                @test [inner(M, A_skewsym, v, w) for v in V, w in V] ≈ I(3)
                @test get_vector(M, A_skewsym, get_coordinates(M, A_skewsym, Y, b), b) ≈ Y
            end
        end
    end

    @testset "Complex skew-Hermitian matrices" begin
        p1 = [1.0im -2.0 -3.0; 2.0 0.0 1.0; 3.0 -1.0 0.0]
        p2 = [2.0im -2.0 - 2.0im -3.0; 2.0 - 2.0im 0.0 -1.0; 3.0 1.0 0.0]
        X = p2 - p1
        Y = ComplexF64[0 im 0; im 0 0; 0 0 -im]
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
                    # the projections take the skew-Hermitian part
                    (project, :Point) => ComplexF64[0 -1 -2; 1 0 -1; 2 1 0],
                    (project, :Vector) => ComplexF64[0 -1 -2; 1 0 -1; 2 1 0],
                    repr => "SkewHermitianMatrices(3, ℂ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => X,
                    :IsPointErrors => [DomainError, ManifoldDomainError],
                    :IsVectorErrors => [DomainError, ManifoldDomainError],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
        @test isapprox(M_complex, -p1, exp(M_complex, p1, log(M_complex, p1, -p1)))
    end

    @testset "Basics" begin
        @test Manifolds.allocation_promotion_function(M_complex, get_vector, ()) === complex
        @test base_manifold(M) === M
        @test riemann_tensor(M, A_skewsym, A_skewsym, B_skewsym, B_skewsym) == zeros(3, 3)
        @test sectional_curvature(M, A_skewsym, A_skewsym, B_skewsym) === 0.0
        @test typeof(get_embedding(M)) ===
            Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3, 3}}}
        @test check_point(M, B_skewsym) === nothing
        @test check_vector(M, B_skewsym, B_skewsym) === nothing
        @test Weingarten!(M, similar(B_skewsym), B_skewsym, B_skewsym, B_skewsym) ==
            zero(B_skewsym)
        A_skewsym2 = copy(A_skewsym)
        @test A_skewsym2 == project!(M, A_skewsym2, A_skewsym2)
        @test A_skewsym == project(M, A_skewsym, A_skewsym)
    end

    @testset "field parameter" begin
        M = SkewHermitianMatrices(3, ℝ; parameter = :field)
        Mc = SkewHermitianMatrices(3, ℂ; parameter = :field)
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int, Int}}
        @test repr(M) == "SkewSymmetricMatrices(3; parameter=:field)"
        @test repr(Mc) == "SkewHermitianMatrices(3, ℂ; parameter=:field)"
    end
end
