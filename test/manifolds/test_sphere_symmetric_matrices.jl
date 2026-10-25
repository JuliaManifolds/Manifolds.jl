using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "SphereSymmetricMatrices" begin
    @test_throws DomainError SphereSymmetricMatrices(0)
    M = SphereSymmetricMatrices(3)
    M_complex = SphereSymmetricMatrices(3, ℂ)
    E(i, j) = (Z = zeros(3, 3); Z[i, j] = 1; Z)
    # Invalids
    A = [1 2 3; 2 4 -5; 3 -5 6] / norm([1 2 3; 2 4 -5; 3 -5 6])
    B = [1 2; 2 4] / norm([1 2; 2 4]) # wrong dimensions
    C = [-3 -im 4; im 5 im; 4 -im 0] / norm([-3 -im 4; im 5 im; 4 -im 0]) # complex
    D = [1 2 3; 4 5 6; 7 8 9] / norm([1 2 3; 4 5 6; 7 8 9]) # not symmetric
    F = [1 2 3; 2 4 -5; 3 -5 6] # not of unit norm

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, embed_project, exp, flat, geodesic,
        get_embedding, injectivity_radius, inner, inverse_retract, is_flat, is_point,
        is_vector, log, manifold_dimension, mid_point, norm, parallel_transport_direction,
        parallel_transport_to, project, rand, repr, representation_size, retract, sharp,
        shortest_geodesic, vector_transport_direction, vector_transport_to, Weingarten,
        zero_vector,
    ]
    common_expectations = Dict(
        :atol => 1.0e-14,
        :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ParallelTransport(),
        distance => π / 4,
        injectivity_radius => π,
        (injectivity_radius, ProjectionRetraction()) => π / 2,
        is_flat => false,
        norm => π / 4,
        representation_size => (3, 3),
    )

    @testset "Real symmetric matrices" begin
        # p and q are orthogonal to E₂₂, so the geodesic between them is a quarter great circle
        p = E(1, 1)
        q = (E(1, 1) + E(2, 2)) / sqrt(2)
        r = (E(1, 2) + E(2, 1)) / sqrt(2)
        X = π / 4 * E(2, 2) # log(M, p, q)
        Y = E(2, 2) + E(1, 3) + E(3, 1)
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Covectors => [flat(M, p, X)],
                # p and X with added normal parts
                :EmbeddedPoints => [2 .* p .+ (E(1, 2) - E(2, 1))],
                :EmbeddedVectors => [X .+ 3 .* p .+ (E(1, 3) - E(3, 1))],
                :Functions => functions,
                :InvalidPoints => Any[D, B, C, F],
                :InvalidVectors => Any[B, C, F, E(1, 2)],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), ProjectionInverseRetraction(),
                ],
                # multiples of p and skew-symmetric matrices are normal to M
                :NormalVectors => [2 .* p .+ (E(1, 2) - E(2, 1))],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M, q, r), log(M, r, p)],
                :VectorTransportMethods => [
                    ParallelTransport(), ProjectionTransport(), SchildsLadderTransport(),
                ],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    (embed, :Point) => p,
                    (embed, :Vector) => X,
                    exp => q,
                    get_embedding => ArraySphere(3, 3),
                    inner => π / 4,
                    log => X,
                    manifold_dimension => 5,
                    mid_point => cos(π / 8) * E(1, 1) + sin(π / 8) * E(2, 2),
                    # the velocity of the geodesic at q
                    parallel_transport_direction => π / 4 * (E(2, 2) - E(1, 1)) / sqrt(2),
                    parallel_transport_to => π / 4 * (E(2, 2) - E(1, 1)) / sqrt(2),
                    (project, :Point) => p,
                    (project, :Vector) => X,
                    (retract, ExponentialRetraction()) => q,
                    repr => "SphereSymmetricMatrices(3, ℝ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) =>
                        π / 4 * (E(2, 2) - E(1, 1)) / sqrt(2),
                    (vector_transport_to, ProjectionTransport()) => π / 8 * (E(2, 2) - E(1, 1)),
                    Weingarten => -2 * X,
                    :IsPointErrors => [
                        DomainError, ManifoldDomainError, ManifoldDomainError, ManifoldDomainError,
                    ],
                    :IsVectorErrors => [
                        ManifoldDomainError, ManifoldDomainError, ManifoldDomainError, DomainError,
                    ],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
    end

    @testset "Complex Hermitian matrices" begin
        K = (im * E(1, 2) - im * E(2, 1)) / sqrt(2) # Hermitian, of unit norm and orthogonal to p
        p = complex(E(1, 1))
        q = (p + K) / sqrt(2)
        r = complex(E(3, 3))
        X = π / 4 * K # log(M, p, q)
        Y = K + complex(E(2, 2))
        Manifolds.Test.test_manifold(
            M_complex,
            Dict(
                :Covectors => [flat(M_complex, p, X)],
                # p and X with added normal parts
                :EmbeddedPoints => [2 .* p .+ im * E(1, 1) .+ (E(1, 2) - E(2, 1))],
                :EmbeddedVectors => [X .+ 3 .* p .+ im * E(2, 2)],
                :Functions => functions,
                :InvalidPoints => AbstractMatrix[complex(D), B, complex(F)],
                :InvalidVectors => AbstractMatrix[B, complex(F), complex(E(1, 2))],
                :InverseRetractionMethods => [
                    LogarithmicInverseRetraction(), ProjectionInverseRetraction(),
                ],
                # multiples of p and skew-Hermitian matrices are normal to M
                :NormalVectors => [3 .* p .+ im * E(1, 1)],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction(), ProjectionRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M_complex, q, r), log(M_complex, r, p)],
                :VectorTransportMethods => [
                    ParallelTransport(), ProjectionTransport(), SchildsLadderTransport(),
                ],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    (embed, :Point) => p,
                    (embed, :Vector) => X,
                    exp => q,
                    get_embedding => ArraySphere(3, 3; field = ℂ),
                    inner => π / 4,
                    log => X,
                    manifold_dimension => 8,
                    mid_point => cos(π / 8) * p + sin(π / 8) * K,
                    (number_eltype, rand) => ComplexF64,
                    parallel_transport_direction => π / 4 * (K - p) / sqrt(2),
                    parallel_transport_to => π / 4 * (K - p) / sqrt(2),
                    (project, :Point) => p,
                    (project, :Vector) => X,
                    (retract, ExponentialRetraction()) => q,
                    repr => "SphereSymmetricMatrices(3, ℂ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => π / 4 * (K - p) / sqrt(2),
                    (vector_transport_to, ProjectionTransport()) => π / 8 * (K - p),
                    Weingarten => -3 * X,
                    :IsPointErrors => [DomainError, ManifoldDomainError, ManifoldDomainError],
                    :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError, DomainError],
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
    end

    @testset "Basics" begin
        @test base_manifold(M) === M
        @test check_point(M, A) === nothing
        @test check_vector(M, A, zeros(3, 3)) === nothing
        A2 = similar(A)
        @test A == project!(M, A2, A)
        @test A == project(M, A)
    end

    @testset "rand" begin
        for N in [SphereSymmetricMatrices(2, ℂ), SphereSymmetricMatrices(3; parameter = :field)]
            rng = MersenneTwister(4711)
            p = rand(rng, N)
            @test is_point(N, p)
            @test is_vector(N, p, rand(rng, N; vector_at = p))
            q = similar(p)
            rand!(rng, N, q)
            @test is_point(N, q)
        end
    end

    Manifolds.Test.test_manifold(
        SphereSymmetricMatrices(3; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, repr], :Points => [E(1, 1)]),
        # Expectations
        Dict(
            get_embedding => ArraySphere(3, 3; parameter = :field),
            repr => "SphereSymmetricMatrices(3, ℝ; parameter=:field)",
        ),
    )
end
