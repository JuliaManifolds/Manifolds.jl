using LinearAlgebra, Manifolds, ManifoldsBase, Test

@testset "Heisenberg matrices" begin
    M = HeisenbergMatrices(1)

    p1 = [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 1.0]
    p2 = [1.0 4.0 -3.0; 0.0 1.0 3.0; 0.0 0.0 1.0]
    p3 = [1.0 -2.0 1.0; 0.0 1.0 1.1; 0.0 0.0 1.0]

    X1 = [0.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 0.0]
    X2 = [0.0 4.0 -3.0; 0.0 0.0 3.0; 0.0 0.0 0.0]
    X3 = [0.0 -2.0 1.0; 0.0 0.0 1.1; 0.0 0.0 0.0]

    N1 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

    q1 = [0.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 1.0]
    q2 = [1.0 2.0 3.0; 0.0 1.0 -1.0; 1.0 0.0 1.0]
    q3 = [1.0 2.0 3.0; 0.0 1.0 -1.0; 0.0 0.0 2.0]
    q4 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 1.0] #sum last row nonzero (excl last)
    q5 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0] # bottom right not I

    Y1 = [1.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 0.0]
    Y2 = [0.0 2.0 3.0; 1.0 0.0 -1.0; 0.0 0.0 0.0]
    Y3 = [0.0 2.0 3.0; 0.0 0.0 -1.0; 0.0 0.0 2.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                embed,
                get_embedding, get_coordinates, get_vector,
                injectivity_radius,
                is_flat, is_point, is_vector,
                parallel_transport_to,
                rand, repr, representation_size,
                Weingarten,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [[2.0, -1.0, 3.0]], #X1 in DefaultONB
            :InvalidPoints => [q1, q2, q3, q4, q5],
            :InvalidVectors => [Y1, Y2, Y3],
            :NormalVectors => [N1],
            :Points => [p1, p2, p3],
            :Vectors => [X1, X2, X3],
            :VectorTransportMethods => [ParallelTransport()],
        ),
        Dict(
            get_embedding => Euclidean(3, 3),
            injectivity_radius => Inf,
            is_flat => true,
            manifold_dimension => 3,
            repr => "HeisenbergMatrices(1)",
            representation_size => (3, 3),
            Weingarten => zeros(3, 3),
        ),
    )

    Mf = HeisenbergMatrices(1; parameter = :field)
    Manifolds.Test.test_manifold(
        Mf,
        Dict(:Functions => [get_embedding, repr]),
        Dict(get_embedding => Euclidean(3, 3; parameter = :field), repr => "HeisenbergMatrices(1; parameter=:field)")
    )
end
