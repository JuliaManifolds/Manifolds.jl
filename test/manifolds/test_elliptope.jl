using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The Elliptope" begin
    @test_throws DomainError Elliptope(2, 3)
    @test_throws DomainError Elliptope(2, 0)

    M = Elliptope(4, 2)
    p = [1.0 0.0; 0.0 1.0; 1 / sqrt(2) -1 / sqrt(2); 1 / sqrt(2) 1 / sqrt(2)]
    q = [4.0 / 5 3.0 / 5; 3.0 / 5.0 -4.0 / 5.0; 1.0 0.0; 0.0 1.0]
    r = [12.0 / 13.0 5.0 / 13.0; 1.0 0.0; -12.0 / 13.0 5.0 / 13.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0; 0.0 0.0; 0.0 0.0]
    Yamb = [0.3 -0.2; 0.1 0.4; -0.5 0.2; 0.25 -0.15] # a vector in the embedding
    X2 = project(M, p, Yamb)
    Y = project(M, q, Yamb)
    Z = project(M, r, Yamb)
    # Invalids
    pf = [2.0 0.0; 0.0 1.0; 1 / sqrt(2) -1 / sqrt(2); 1 / sqrt(2) 1 / sqrt(2)]
    Xf = [0.1 1.0; 1.0 0.1; 0.0 0.0; 0.0 0.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_retraction_method, default_vector_transport_method, embed,
                get_embedding, inner, is_default_metric, is_flat, is_point, is_vector,
                manifold_dimension, norm, project, rand, repr, representation_size, retract,
                zero_vector,
            ],
            :Points => [p, q, r], :Vectors => [X, Y, Z], :SecondVector => X2,
            :EmbeddedPoints => [2 .* p], :EmbeddedVectors => [Yamb],
            :InvalidPoints => [pf], :InvalidVectors => [Xf],
            :InverseRetractionMethods => [missing],
            :RetractionMethods => [ProjectionRetraction()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            default_retraction_method => ProjectionRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            embed => p,
            (embed, :Vector) => X,
            get_embedding => Euclidean(4, 2),
            inner => dot(X, X2),
            is_default_metric => EuclideanMetric(),
            is_flat => false,
            manifold_dimension => 3,
            norm => sqrt(2),
            (project, :Point) => p,
            (project, :Vector) => X2,
            repr => "Elliptope(4, 2)",
            representation_size => (4, 2),
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
        ),
    )

    # tested separately, since the default `direction` requires `log`, which is not available
    Manifolds.Test.test_vector_transport(
        M, p, X, q, ProjectionTransport();
        direction = nothing,
        expected_value = project(M, q, X),
        name = "vector_transport_to(M, p, X, q, ProjectionTransport())",
        atol = 1.0e-14,
    )

    @testset "Basics" begin
        @test base_manifold(M) === M
        @test metric(M) == EuclideanMetric()
        @test riemannian_gradient(M, p, Yamb) == X2
        @test isapprox(X2, project(M, p, X2); atol = 1.0e-14) # the projection is idempotent
        @test isapprox(p' * X2, (p' * X2)'; atol = 1.0e-14) # and lands in the horizontal space
    end

    @testset "tangent vectors of any length" begin
        M2 = Elliptope(3, 2)
        p2 = [1.0 0.0; 0.0 1.0; 1 / sqrt(2) 1 / sqrt(2)]
        @test is_vector(M2, p2, project(M2, p2, [100.0 -50.0; 30.0 20.0; -10.0 5.0]))
        @test !is_vector(M2, p2, project(M2, p2, [1.0 -0.5; 0.3 0.2; -0.1 0.4]) .+ 1.0e-6 .* p2)
    end

    Manifolds.Test.test_manifold(
        Elliptope(4, 2; parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, is_vector, manifold_dimension, repr],
            :Points => [p, q], :Vectors => [X, Y],
        ),
        # Expectations
        Dict(
            get_embedding => Euclidean(4, 2; parameter = :field),
            manifold_dimension => 3,
            repr => "Elliptope(4, 2; parameter=:field)",
        ),
    )
end
