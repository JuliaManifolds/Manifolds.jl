using LinearAlgebra, Manifolds, ManifoldsBase, StaticArrays, Test

@testset "Lorentz manifold" begin
    @test_throws DomainError Lorentz(-1)

    M = Lorentz(3)
    p = [0.0, 0.0, 0.0]
    q = [1.0, 2.0, 2.0]
    r = [1.0, -1.0, 0.5]
    X = [1.0, 2.0, 3.0]
    Y = [2.0, 3.0, 4.0]
    Z = [0.0, 1.0, -1.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_retraction_method, exp, inner, inverse_retract,
                is_flat, is_point, is_positive_definite_metric, is_vector, log,
                manifold_dimension, metric_signature, parallel_transport_to, repr,
                representation_size, retract, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => [[1.0, 2.0]],
            :InvalidVectors => [[1.0, 2.0]],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Points => [p, q, r],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Y, Z],
            :VectorTransportMethods => [ParallelTransport()],
        ),
        # Expectations
        Dict(
            default_retraction_method => ExponentialRetraction(),
            # geodesics are straight lines
            exp => X,
            # the time component enters with a negative sign, 2 + 6 - 12
            inner => -4.0,
            (inverse_retract, LogarithmicInverseRetraction()) => q,
            is_flat => true,
            # one time dimension
            is_positive_definite_metric => false,
            log => q,
            manifold_dimension => 3,
            metric_signature => (2, 1, 0),
            parallel_transport_to => X,
            repr => "MetricManifold(Euclidean(3; field=ℝ), MinkowskiMetric())",
            representation_size => (3,),
            (retract, ExponentialRetraction()) => X,
            (vector_transport_to, ParallelTransport()) => X,
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
        ),
    )

    @testset "Minkowski metric" begin
        @test M == MetricManifold(Euclidean(3), MinkowskiMetric())
        @test local_metric(M, p) == Diagonal([1.0, 1.0, -1.0])
        @test minkowski_metric(X, Y) == -4.0
        @test minkowski_metric(SVector(1.0, 2.0, 3.0), SVector(2.0, 3.0, 4.0)) == -4.0
        # lightlike vectors have length zero
        @test inner(M, p, [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]) == 0.0
        @test inner(M, p, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]) == -1.0
        @test parallel_transport_direction(M, p, X, q) == X
        Yd = similar(X)
        parallel_transport_direction!(M, Yd, p, X, q)
        @test Yd == X
    end

    @testset "Signatures of manifolds built from the Lorentz manifold" begin
        # each copy of the Lorentz manifold contributes one time dimension
        @test metric_signature(ProductManifold(M, Sphere(2))) == (4, 1, 0)
        @test metric_signature(PowerManifold(M, 2, 3)) == (12, 6, 0)
        @test metric_signature(PowerManifold(M, NestedPowerRepresentation(), 2)) == (4, 2, 0)
        # the base and the fiber of the tangent and cotangent bundles
        @test metric_signature(TangentSpace(M, p)) == (2, 1, 0)
        @test metric_signature(CotangentSpace(M, p)) == (2, 1, 0)
        @test metric_signature(TangentBundle(M)) == (4, 2, 0)
        @test metric_signature(CotangentBundle(M)) == (4, 2, 0)
        @test metric_signature(ValidationManifold(M)) == (2, 1, 0)
        @test !is_positive_definite_metric(ProductManifold(M, Sphere(2)))
        # products and powers of Riemannian manifolds are Riemannian
        @test metric_signature(ProductManifold(Sphere(2), Euclidean(2))) == (4, 0, 0)
        @test metric_signature(Torus(3)) == (3, 0, 0)
        @test is_positive_definite_metric(TangentBundle(Sphere(2)))
    end

    Mf = Lorentz(3; parameter = :field)
    Manifolds.Test.test_manifold(
        Mf,
        Dict(
            :Functions => [
                inner, is_flat, is_point, is_positive_definite_metric, manifold_dimension,
                metric_signature, repr,
            ],
            :Points => [p],
            :SecondVector => Y,
            :Vectors => [X],
        ),
        # Expectations
        Dict(
            inner => -4.0,
            is_flat => true,
            is_positive_definite_metric => false,
            manifold_dimension => 3,
            metric_signature => (2, 1, 0),
            repr => "MetricManifold(Euclidean(3; field=ℝ, parameter=:field), MinkowskiMetric())",
        ),
    )
    @test local_metric(Mf, p) == Diagonal([1.0, 1.0, -1.0])
end
