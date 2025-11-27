using Manifolds, Test, LinearAlgebra

@testset "Cholesky Space" begin
    M = CholeskySpace(3)

    A(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]

    p1 = cholesky([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1]).L
    p2 = cholesky([2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1]).L
    p3 = cholesky(A(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1] * transpose(A(π / 6))).L

    # Tangent vectors are symmetric matrices
    X1 = [3.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
    X2 = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0]

    q1 = zeros(2, 3) # wrong size
    q2 = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0] # nonpos diag
    q3 = [2.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 4.0] # no lower and nonsym

    Y = [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0] # not symmetric
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                default_vector_transport_method, distance, exp,
                inner, is_flat, is_point, is_vector,
                get_coordinates, get_vector,
                log, manifold_dimension,
                parallel_transport_to, rand, repr, representation_size,
                zero_vector,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            :Points => [p1, p2, p3],
            :Vectors => [X1, X2],
            :InvalidPoints => [q1, q2, q3],
            :InvalidVectors => [Y],
            :VectorTransportMethods => [
                ParallelTransport(),
                SchildsLadderTransport(),
                PoleLadderTransport(),
            ],
        ),
        Dict(
            default_vector_transport_method => ParallelTransport(),
            is_flat => true,
            manifold_dimension => 6,
            repr => "CholeskySpace(3)",
            representation_size => (3, 3),
        )
    )

    M = CholeskySpace(3; parameter = :field)
    Manifolds.Test.test_manifold(
        M,
        Dict(:Functions => [repr]),
        Dict(repr => "CholeskySpace(3; parameter=:field)")
    )
end
