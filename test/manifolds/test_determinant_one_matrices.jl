using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test


@testset "Matrices of Determinant 1" begin
    # Real case
    M = DeterminantOneMatrices(2)
    p = [1.0 0.0; 0.0 1.0]
    q = [10.0 0.0; 0.0 0.1]
    pf = [2.0 0.0; 0.0 1.0]
    qf = [1.0 0.0; 0.0 0.0]
    X = [0.0 0.23; -0.14 0.0]
    Y = [0.0 -0.5; 0.5 0.0]
    Xf = [1.0 1.1; 1.2 0.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [is_point, is_vector, project, embed],
            :Points => [p, q], :Vectors => [X, Y],
            :InvalidPoints => [pf, qf], :InvalidVectors => [Xf],
            :EmbeddedPoints => [p], :EmbeddedVectors => [X],
        ),
        # Expectations
        Dict(
            :manifold_dimension => 3,
            repr => "DeterminantOneMatrices(2, ?)",
            get_embedding => Euclidean(2, 2),
        ),
    )
    # Complex case
    Mc = DeterminantOneMatrices(2, ℂ)
    pc = [1.0im 0.0; 0.0 -1.0im]
    qc = [10.0im 0.0; 0.0 -0.1im]
    Xc = [0.0 0.23im; -0.14 0.0]
    Yc = [0.0 -0.5im; 0.5im 0.0]
    pcf = [10.0 0.0; 0.0 0.0]
    qcf = [2.0 0.0; 0.0 1.0]
    Xf = [1.0 + 2.0im 1.1 + 0.1im; 1.2 0.0]

    Manifolds.Test.test_manifold(
        Mc,
        Dict(
            :Functions => [is_point, is_vector, project, embed],
            :Points => [pc, qc], :Vectors => [Xc, Yc],
            :InvalidPoints => [pcf, qcf], :InvalidVectors => [Xf],
            :EmbeddedPoints => [pc], :EmbeddedVectors => [Xc],
        ),
        # Expectations
        Dict(
            :manifold_dimension => 3,
            repr => "DeterminantOneMatrices(2, ℂ)",
            get_embedding => Euclidean(2, 2; field = ℂ),
        ),
    )


    # Parameter
    Manifolds.Test.test_manifold(
        DeterminantOneMatrices(2; parameter = :field),
        Dict(:Functions => [repr]),
        Dict(repr => "DeterminantOneMatrices(2, ℂ; parameter=:field)"),
    )
end