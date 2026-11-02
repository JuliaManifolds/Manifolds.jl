using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The oblique manifold" begin
    @test_throws DomainError Oblique(0, 2)
    @test_throws DomainError Oblique(2, -1)

    M = Oblique(3, 2)
    # columns are points on the sphere
    p1 = [1.0 1.0; 0.0 0.0; 0.0 0.0]
    p2 = [1.0 1 / sqrt(2); 0.0 1 / sqrt(2); 0.0 0.0]
    p3 = [1 / sqrt(2) 1.0; 1 / sqrt(2) 0.0; 0.0 0.0]
    X = [0.0 0.0; 0.0 π / 4; 0.0 0.0]
    X2 = [0.0 0.0; 0.5 0.0; 0.3 0.1]
    Y = [0.0 0.0; 0.0 0.0; 0.2 -0.3]
    W = [0.3 -0.2; 0.5 0.0; 0.3 0.1] # an ambient vector at p1
    # Invalids
    pf1 = [2.0 2.0; 0.0 0.0; 0.0 0.0] # columns not of unit norm
    pf2 = [2.0, 0.0, 0.0] # wrong size
    Xf1 = [1.0 0.0; 0.0 0.0; 0.0 0.0] # first column not orthogonal to p1
    Xf2 = [0.0, 0.0, 0.0] # wrong size
    bases = [DefaultOrthonormalBasis()]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => bases,
            :Coordinates => [[0.5, 0.3, 0.0, 0.1]],
            :Covectors => [Manifolds.RieszRepresenterCotangentVector(M, p1, X)],
            :EmbeddedPoints => [p1],
            :EmbeddedVectors => [W],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, embed, exp, flat, geodesic,
                get_basis, get_coordinates, get_embedding, get_vector, injectivity_radius,
                inner, inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension,
                mid_point, norm, parallel_transport_direction, parallel_transport_to, project,
                rand, repr, representation_size, retract, sharp, shortest_geodesic,
                vector_transport_direction, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => AbstractArray[pf1, pf2],
            :InvalidVectors => AbstractArray[Xf1, Xf2],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Points => [p1, p2, p3],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => X2,
            :Vectors => [X, project(M, p2, Y), project(M, p3, X2)],
            :VectorTransportMethods => [ParallelTransport()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-11, shortest_geodesic => 1.0e-11),
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => π / 4,
            exp => p2,
            (embed, :Point) => p1,
            (embed, :Vector) => X,
            (get_coordinates, bases[1]) => [0.0, 0.0, π / 4, 0.0],
            (get_vector, [0.5, 0.3, 0.0, 0.1], bases[1]) => X2,
            get_embedding => Euclidean(3, 2; parameter = :field),
            injectivity_radius => π,
            inner => 0.0,
            is_flat => false,
            log => X,
            manifold_dimension => 4,
            norm => π / 4,
            (project, :Point) => p1,
            (project, :Vector) => [0.0 0.0; 0.5 0.0; 0.3 0.1],
            repr => "Oblique(3, 2, ℝ)",
            representation_size => (3, 2),
            sharp => X,
            :IsPointErrors => [CompositeManifoldError, DomainError],
            :IsVectorErrors => [ComponentManifoldError{Int64, DomainError}, DomainError],
            :IsVectorBasepointError => CompositeManifoldError,
        ),
    )

    @testset "Sphere powers" begin
        @test Sphere(2)^3 === Oblique(3, 3)
        @test Sphere(2)^(3,) === PowerManifold(Sphere(2), 3)
        @test ^(Sphere(2), 2) === Oblique(3, 2)
        @test typeof(^(Sphere(2), 2)) ==
            Oblique{ℝ, ManifoldsBase.TypeParameter{Tuple{3, 2}}, ManifoldsBase.TypeParameter{Tuple{2}}}
        @test power_dimensions(M) == power_dimensions(PowerManifold(Sphere(2), 2)) == (2,)
    end

    @testset "field parameter" begin
        @test typeof(^(Sphere(2; parameter = :field), 2)) ==
            Oblique{ℝ, Tuple{Int, Int}, Tuple{Int}}
        M2 = Oblique(3, 2; parameter = :field)
        @test repr(M2) == "Oblique(3, 2, ℝ; parameter=:field)"
        @test get_embedding(M2) == Euclidean(3, 2; parameter = :field)
    end
end
