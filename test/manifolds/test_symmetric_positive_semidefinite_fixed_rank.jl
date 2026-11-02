using LinearAlgebra, Manifolds, ManifoldsBase, Test

@testset "Symmetric positive semidefinite matrices of fixed rank" begin
    @test_throws DomainError SymmetricPositiveSemidefiniteFixedRank(2, 3)

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, flat, geodesic, get_embedding,
        inner, inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension,
        mid_point, norm, repr, representation_size, retract, sharp, shortest_geodesic,
        vector_transport_direction, vector_transport_to, zero_vector,
    ]
    common_expectations = Dict(
        :atol => 1.0e-14,
        :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ProjectionTransport(),
        is_flat => false,
        representation_size => (4, 2),
        :IsPointErrors => [DomainError],
        :IsVectorErrors => [ManifoldDomainError],
        :IsVectorBasepointError => ManifoldDomainError,
    )

    @testset "Real matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4, 2)
        q = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        # qᵀq2 is symmetric positive definite, so q2 is the representative closest to q
        # and the logarithmic map is the difference
        q2 = [2.0 0.0; 0.0 1.0; 1.0 0.0; 0.0 1.0]
        q3 = [0.0 0.0; 1.0 0.0; 0.0 1.0; 0.0 0.0]
        X = q2 - q
        Y = [0.0 1.0; 1.0 0.0; 0.0 2.0; 1.0 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Covectors => [flat(M, q, X)],
                :Functions => functions,
                :InvalidPoints => [[1.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]], # of rank one
                :InvalidVectors => [ones(3, 3)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [q, q2, q3],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M, q2, q3), log(M, q3, q)],
                :VectorTransportMethods => [ProjectionTransport()],
            ),
            merge(
                common_expectations,
                Dict(
                    distance => sqrt(3),
                    (embed, :Point) => q,
                    (embed, :Vector) => X,
                    exp => q2,
                    get_embedding => Euclidean(4, 2),
                    inner => dot(X, Y),
                    log => X,
                    manifold_dimension => 7,
                    mid_point => (q + q2) / 2,
                    norm => sqrt(3),
                    repr => "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℝ)",
                    sharp => X,
                    # q2ᵀX is symmetric, so X is already horizontal at q2
                    (vector_transport_to, ProjectionTransport()) => X,
                ),
            ),
        )

        @testset "Basics" begin
            @test is_point(M, [1.0 0.0; 0.0 1.0e-5; 0.0 0.0; 0.0 0.0])
            @test metric(M) == EuclideanMetric()
            Yamb = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
            Z = project(M, q, Yamb)
            @test Z ≈ [1.0 2.5; 2.5 4.0; 5.0 6.0; 7.0 8.0]
            # the projection removes the vertical part, so that qᵀZ is symmetric
            @test q' * Z ≈ (q' * Z)'
            @test riemannian_gradient(M, q, Yamb) == Z
            # also where qᵀq is not the identity
            Z2 = project(M, q2, Yamb)
            @test q2' * Z2 ≈ (q2' * Z2)'
            @test project(M, q2, Z2) ≈ Z2
            Z3 = vector_transport_to(M, q, Z, q3, ProjectionTransport())
            @test is_vector(M, q3, Z3)
            @test isapprox(M, q3, Z3, project(M, q3, Z))
        end
    end

    @testset "Complex matrices" begin
        M = SymmetricPositiveSemidefiniteFixedRank(4, 2, ℂ)
        q = ComplexF64[1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]
        q2 = ComplexF64[2.0 0.0; 0.0 1.0; im 0.0; 0.0 1.0]
        q3 = ComplexF64[0.0 0.0; 1.0 0.0; 0.0 im; 0.0 0.0]
        X = q2 - q
        Y = ComplexF64[0.0 im; 1.0 0.0; 0.0 2.0; im 0.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Covectors => [flat(M, q, X)],
                :Functions => functions,
                :InvalidPoints => [ComplexF64[1.0 0.0; im 0.0; 0.0 0.0; 0.0 0.0]], # of rank one
                :InvalidVectors => [ones(ComplexF64, 3, 3)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [q, q2, q3],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, log(M, q2, q3), log(M, q3, q)],
                :VectorTransportMethods => [ProjectionTransport()],
            ),
            merge(
                common_expectations,
                Dict(
                    distance => sqrt(3),
                    (embed, :Point) => q,
                    (embed, :Vector) => X,
                    exp => q2,
                    get_embedding => Euclidean(4, 2; field = ℂ),
                    inner => dot(X, Y),
                    log => X,
                    manifold_dimension => 12,
                    mid_point => (q + q2) / 2,
                    norm => sqrt(3),
                    repr => "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℂ)",
                    sharp => X,
                    (vector_transport_to, ProjectionTransport()) => X,
                ),
            ),
        )

        @testset "the projection gives horizontal vectors" begin
            Yamb = ComplexF64[1.0 2.0im; 3.0 4.0; 5.0im 6.0; 7.0 8.0]
            Z = project(M, q2, Yamb)
            @test q2' * Z ≈ (q2' * Z)'
            @test project(M, q2, Z) ≈ Z
        end

        @testset "exp and log only agree up to the equivalence class" begin
            qc = ComplexF64[1.0 0.0; 0.0 1.0im; 0.0 0.0; 0.0 0.0]
            qc2 = ComplexF64[2.0 1.0; 0.0 1.0; 0.0 1.0im; 0.0 0.0]
            qe = exp(M, qc, log(M, qc, qc2))
            @test qe * qe' ≈ qc2 * qc2'
            @test isapprox(M, qe, qc2; atol = 5 * eps(Float64))
        end
    end

    Manifolds.Test.test_manifold(
        SymmetricPositiveSemidefiniteFixedRank(4, 2; parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, manifold_dimension, repr],
            :Points => [[1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0]],
        ),
        # Expectations
        Dict(
            get_embedding => Euclidean(4, 2; parameter = :field),
            manifold_dimension => 7,
            repr => "SymmetricPositiveSemidefiniteFixedRank(4, 2, ℝ; parameter=:field)",
        ),
    )
end
