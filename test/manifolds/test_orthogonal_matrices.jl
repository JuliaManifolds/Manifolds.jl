using LinearAlgebra, Manifolds, Quaternions, Test

@testset "Orthogonal Matrices" begin
    M = OrthogonalMatrices(3)
    α = 0.1
    p1 = [cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    p2 = [-cos(α) sin(α) 0.0; -sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    p3 = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]
    pE = ones(3, 3)
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                default_vector_transport_method,
                get_embedding,
                injectivity_radius, is_flat,
                rand, repr,
            ],
            :EmbeddedPoints => [p3],
            :Points => [p1, p2, p3],
            :RetractionMethods => [PolarRetraction()],
        ),
        Dict(
            default_vector_transport_method => ProjectionTransport(),
            get_embedding => Euclidean(3, 3),
            injectivity_radius => π * sqrt(2),
            (injectivity_radius, PolarRetraction()) => π / sqrt(2),
            is_flat => false,
            manifold_dimension => 3,
            repr => "OrthogonalMatrices(3)",
        )
    )
end
