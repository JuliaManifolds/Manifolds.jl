using Manifolds, Test

Test.@testset "Centered Matrices" begin

    M = CenteredMatrices(3, 2)

    p1 = [1 2; 4 5; -5 -7]
    p2 = [0 0; 1 -1; -1 1]
    p3 = [0.5 1; -1 -0.7; 0.5 -0.3]
    q1 = [1 2 3; 4 5 6; -5 -7 -9]    #wrong dimensions
    q2 = [-3 -im; 2 im; 1 0]         #complex
    q3 = [1 2; 3 4; 5 6]             #not centered

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                is_point, is_vector, is_flat,
                get_embedding,
                manifold_dimension,
                project,
                repr,
            ],
            :Points => [p1, p2, p3],
            :Vectors => [p1],
            :EmbeddedPoints => [p1],
            :InvalidPoints => [q1, q2, q3],
            :InvalidVectors => [p1, q2, q3],

        ),
        Dict(
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
            manifold_dimension => 4,
        )
    )
end
