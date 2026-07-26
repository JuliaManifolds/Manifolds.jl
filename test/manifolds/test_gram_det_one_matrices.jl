using Manifolds, Test

@testset "Matrices of Gramian Determinant One" begin
    M = GramianDetOneMatrices(3, 2)
    p1 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    p2 = [0.0 1.0; 1.0 0.0; 0.0 0.0]
    X1 = [0.0 1.0; 0.0 0.0; 0.0 0.0]
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                embed,
                get_embedding, # TODO: get_coordinates, get_vector,
                repr, representation_size,
            ],
            :Bases => [DefaultOrthonormalBasis()],
            # TODO: Define invalid points and vectors
            # :InvalidPoints => [q1, q2, q3, q4, q5],
            # :InvalidVectors => [Y1, Y2, Y3],
            :Points => [p1, p2],
            :Vectors => [X1],
        ),
        Dict(
            get_embedding => Euclidean(3, 2),
            manifold_dimension => 5,
            repr => "GramianDetOneMatrices(3, 2)",
            representation_size => (3, 2),
        ),
    )
end