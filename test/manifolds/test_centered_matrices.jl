using Manifolds, Test

Test.@testset "Centered Matrices" begin

    M = CenteredMatrices(3, 2)

    p1 = [1 2; 4 5; -5 -7]
    p2 = [0 0; 1 -1; -1 1]
    p3 = [0.5 1; -1 -0.7; 0.5 -0.3]
    q1 = [1 2 3; 4 5 6; -5 -7 -9]    #wrong dimensions
    q2 = [-3 -im; 2 im; 1 0]         #complex
    q3 = [1.0 2; 3 4; 5 6]             #not centered

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                embed,
                get_embedding,
                is_point, is_vector, is_flat,
                manifold_dimension,
                project,
                repr, representation_size,
            ],
            :Points => [p1, p2, p3],
            :Vectors => [p1],
            :EmbeddedPoints => [p1],
            :InvalidPoints => Matrix[q1, q2, q3], #To avoid implicit conversion to complex matrices
            :InvalidVectors => Matrix[q1, q2, q3],
        ),
        Dict(
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, DomainError],
            :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError, DomainError],
            is_flat => true,
            get_embedding => Euclidean(3, 2),
            manifold_dimension => 4,
            repr => "CenteredMatrices(3, 2, ℝ)",
            representation_size => (3, 2),
        )
    )
    Mf = CenteredMatrices(3, 2; parameter = :field)
    Manifolds.Test.test_manifold(
        Mf,
        Dict(:Functions => [repr, get_embedding]),
        Dict(
            repr => "CenteredMatrices(3, 2, ℝ; parameter=:field)",
            get_embedding => Euclidean(3, 2; parameter = :field),
        )
    )

    Mc = CenteredMatrices(3, 2, ℂ)
    p4 = q2
    p5 = [1.0 1.0im; -1.0im 0.0; -1.0 + 1.0im -1.0im]
    p6 = [1.0im 0.0; -2.0im 1.0im; 1.0im -1.0im]
    # Complex case
    Manifolds.Test.test_manifold(
        Mc,
        Dict(
            :Functions => [
                embed,
                get_embedding,
                is_point, is_vector, is_flat,
                manifold_dimension,
                project,
                repr, representation_size,
            ],
            :Points => [p4, p5, p6],
            :Vectors => [p4],
            :EmbeddedPoints => [p4],
        ),
        Dict(
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
            is_flat => true,
            get_embedding => Euclidean(3, 2; field = ℂ),
            manifold_dimension => 8,
            repr => "CenteredMatrices(3, 2, ℂ)",
            representation_size => (3, 2),
        )
    )

end
