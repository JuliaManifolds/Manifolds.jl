using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Torus" begin
    @test_throws DomainError Torus(-1)
    @test_throws DomainError Base.:^(Circle(), -1)

    M = Torus(2)
    # the angles of p and q differ by 2π - 5.5 in the first coordinate, across ±π
    p = [3.0, 2.0]
    q = [-2.5, 2.0]
    r = [0.0, -1.0]
    X = [0.5, -1.0]
    Y = [2.0, 0.5]
    Z = [0.0, 3.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [DefaultOrthonormalBasis()],
            :Coordinates => [Y],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, exp, geodesic, get_basis,
                get_coordinates, get_vector, get_vectors, injectivity_radius, inner, inverse_retract, is_flat,
                is_point, is_vector, log, manifold_dimension, mid_point, norm,
                parallel_transport_to, rand, repr, representation_size, retract,
                shortest_geodesic, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => [9.0, [9.0, 9.0], [9.0 9.0]],
            :InvalidVectors => [0.0, [0.0, 0.0, 0.0]],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Points => [p, q, r],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Y, Z],
            :VectorTransportMethods => [ParallelTransport()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-12, shortest_geodesic => 1.0e-12),
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => 2π - 5.5,
            # the angles are taken in [-π, π)
            exp => [3.5 - 2π, 1.0],
            (get_coordinates, DefaultOrthonormalBasis()) => X,
            (get_vector, Y, DefaultOrthonormalBasis()) => Y,
            (get_vectors, DefaultOrthonormalBasis()) => :Orthonormal,
            injectivity_radius => π,
            inner => 0.5,
            (inverse_retract, LogarithmicInverseRetraction()) => [2π - 5.5, 0.0],
            is_flat => true,
            log => [2π - 5.5, 0.0],
            manifold_dimension => 2,
            mid_point => [0.25 - π, 2.0],
            norm => sqrt(1.25),
            parallel_transport_to => X,
            repr => "Torus(2)",
            representation_size => (2,),
            (retract, ExponentialRetraction()) => [3.5 - 2π, 1.0],
            (vector_transport_to, ParallelTransport()) => X,
            :IsPointErrors => [DomainError, CompositeManifoldError, CompositeManifoldError],
            :IsVectorErrors => [DomainError, DomainError],
            :IsVectorBasepointError => DomainError,
        ),
    )

    # the charts of the default atlas are the angles, so the coordinates in the induced basis
    # are the components of tangent vectors
    A = get_default_atlas(M)
    B = induced_basis(M, A, get_chart_index(M, A, p))
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [B],
            :Coordinates => [Y],
            :Functions => [get_coordinates, get_vector],
            :Points => [p],
            :Vectors => [X],
        ),
        # Expectations
        Dict((get_coordinates, B) => X, (get_vector, Y, B) => Y),
    )

    @testset "Powers of the circle" begin
        @test Circle()^3 === Torus(3)
        @test ^(Circle(), 2) === Torus(2)
        @test typeof(^(Circle(), 2)) == Torus{2}
        @test Circle()^(3,) === PowerManifold(Circle(), 3)
        @test Circle(ℂ)^3 === PowerManifold(Circle(ℂ), 3)
    end
end
