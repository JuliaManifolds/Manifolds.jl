using LinearAlgebra, Manifolds, StaticArrays, Test

@testset "The Stiefel manifolds" begin
    M = Stiefel(3, 2)
    p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    q = 1 ./ sqrt(2) .* [ 1.0 -1.0; 1.0 1.0; 0.0 0.0]
    r = 1 ./ sqrt(2) .* [ 0.0 0.0; 1.0 -1.0; 1.0 1.0]
    X = [0.0 0.0; 0.0 0.0; -0.1 0.2]
    Y = [0.0 0.0; 0.0 0.0; -0.3 0.4]
    Z = [0.5 -0.6; 0.0 0.0; 0.0 0.0]

    # Embedded
    ep = [1.0 0.0; 0.0 2.0; 0.0 0.0]
    @testset "Basics" begin
        @test base_manifold(M) === M
    end
    @testset "gradient and metric conversion" begin
        Y = change_metric(M, EuclideanMetric(), p, X)
        @test Y == X
        Z = change_representer(M, EuclideanMetric(), p, X)
        @test Z == X
        # In this case it stays as is
        @test riemannian_Hessian(M, p, Y, Z, X) == Z
        V = [1.0 1.0; 0.0 0.0; 0.0 0.0] # From T\bot_pM.
        @test Weingarten(M, p, X, V) == [0.0 0.0; 0.0 0.0; 0.1 0.1]
    end
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :EmbeddedPoints => [ep],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, embed, embed_project, exp,
                geodesic, get_basis, get_coordinates, get_embedding, get_vector, get_vectors,
                injectivity_radius,
                inner, inverse_retract, is_default_metric, is_flat, is_point, is_vector, log,
                manifold_dimension, mid_point, norm, project, rand, repr, representation_size, retract,
                shortest_geodesic, vector_transport_direction, vector_transport_to, zero_vector,
            ],
            :InverseRetractionMethods => [PolarInverseRetraction(), PolarLightInverseRetraction(), ProjectionInverseRetraction(), QRInverseRetraction()],
            :Points => [p, q, r],
            :Vectors => [X, Y, Z],
            :RetractionMethod => [CayleyRetraction(), PadeRetraction(2), PolarRetraction(), PolarLightRetraction(), ProjectionRetraction(), QRRetraction()],
            :VectorTransportMethods => [
                DifferentiatedRetractionVectorTransport(PolarRetraction()),
                DifferentiatedRetractionVectorTransport(QRRetraction()),
                # errors on alias test
                # ProjectionTransport(),
            ],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-12,
            default_inverse_retraction_method => PolarInverseRetraction(),
            default_retraction_method => PolarRetraction(),
            default_vector_transport_method => DifferentiatedRetractionVectorTransport(PolarRetraction()),
            injectivity_radius => π,
            manifold_dimension => 3,
        )
    )
    @testset "Stiefel(2, 1) special case" begin
        M21 = Stiefel(2, 1)
        w = inverse_retract(
            M21, SMatrix{2, 1}([0.0, 1.0]), SMatrix{2, 1}([sqrt(2), sqrt(2)]),
            QRInverseRetraction(),
        )
        @test isapprox(M21, w, SMatrix{2, 1}([1.0, 0.0]))
    end
    # TODO: Complex case
end
