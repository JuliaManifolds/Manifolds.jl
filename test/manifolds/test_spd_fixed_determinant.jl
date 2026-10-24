using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Isochoric matrices" begin
    @test_throws DomainError SPDFixedDeterminant(0)
    M = SPDFixedDeterminant(2, 1.0)
    # the manifold is totally geodesic in the symmetric positive definite matrices, and
    # diagonal matrices of determinant one form a geodesic through the identity
    a = 0.5
    p = [1.0 0.0; 0.0 1.0]
    q = [exp(a) 0.0; 0.0 exp(-a)]
    r = [2.0 1.0; 1.0 1.0]
    X = [a 0.0; 0.0 -a] # log(M, p, q)
    Y = [1.0 2.0; 2.0 -1.0]
    Xq = [a*exp(a) 0.0; 0.0 -a*exp(-a)] # the velocity of the geodesic at q

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Covectors => [flat(M, p, X)],
            :EmbeddedPoints => [2 .* p],
            # multiples of p are normal to M
            :EmbeddedVectors => [X .+ 3 .* p],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, embed, embed_project, exp, flat,
                geodesic, get_embedding, injectivity_radius, inner, inverse_retract, is_flat,
                is_point, is_vector, log, manifold_dimension, mid_point, norm,
                parallel_transport_direction, parallel_transport_to, project, rand, repr,
                representation_size, retract, sharp, shortest_geodesic,
                vector_transport_direction, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => [2 .* p, [1.0 0.0; 0.0 -1.0]],
            :InvalidVectors => [[1.0 0.1; 0.1 1.0], [0.0 1.0; 0.0 0.0]],
            :InverseRetractionMethods => [LogarithmicInverseRetraction()],
            :Points => [p, q, r],
            :RetractionMethods => [ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, log(M, q, r), log(M, r, p)],
            :VectorTransportMethods => [ParallelTransport(), ProjectionTransport()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ParallelTransport(),
            distance => sqrt(2) * a,
            (embed, :Point) => p,
            (embed, :Vector) => X,
            exp => q,
            get_embedding => SymmetricPositiveDefinite(2),
            injectivity_radius => Inf,
            # the affine invariant metric at the identity
            inner => tr(X * Y),
            is_flat => false,
            log => X,
            manifold_dimension => 2,
            mid_point => [exp(a / 2) 0.0; 0.0 exp(-a / 2)],
            norm => sqrt(2) * a,
            parallel_transport_direction => Xq,
            parallel_transport_to => Xq,
            (project, :Point) => p,
            (project, :Vector) => X,
            (retract, ExponentialRetraction()) => q,
            repr => "SPDFixedDeterminant(2, 1.0)",
            representation_size => (2, 2),
            sharp => X,
            (vector_transport_to, ParallelTransport()) => Xq,
            # removes the multiple of q that makes the trace of q⁻¹X vanish
            (vector_transport_to, ProjectionTransport()) => X + a * sinh(a) * q,
            :IsPointErrors => [DomainError, ManifoldDomainError],
            :IsVectorErrors => [DomainError, ManifoldDomainError],
            :IsVectorBasepointError => ManifoldDomainError,
        ),
    )

    @testset "Basics" begin
        # Determinant is 1.00020001
        @test !is_point(M, 1.0001 .* p)
        @test is_point(M, 1.0001 .* p; atol = 1.0e-2)
        @test project(M, p, [1.0 0.1; 0.1 1.0]) == [0.0 0.1; 0.1 0.0]
        # the geodesics agree with those of the embedding
        @test distance(M, q, exp(get_embedding(M), p, X)) ≈ 0 atol = 6.0e-16
        @test norm(M, p, log(get_embedding(M), p, q) - X) ≈ 0 atol = 3.0e-16
        @test riemann_tensor(M, p, X, Y, X) ≈ riemann_tensor(get_embedding(M), p, X, Y, X)
        M4 = SPDFixedDeterminant(2, 4.0)
        @test project(M4, p) ≈ 2 .* p
        @test is_point(M4, 2 .* p)
    end

    @testset "tangent vectors away from the identity" begin
        M3 = SPDFixedDeterminant(3, 1.0)
        p3 = [2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.5]
        # trace zero, but the geodesic in this direction changes the determinant
        X3 = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]
        @test !is_vector(M3, p3, X3)
        @test project(M3, p3, X3) ≈ [4.0 0.0 0.0; 0.0 -2.5 0.0; 0.0 0.0 0.25] ./ 3
        q3 = project(M3, [4.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 2.0])
        Z3 = [1.0 2.0 3.0; 2.0 -1.0 0.5; 3.0 0.5 4.0]
        @test is_vector(M3, q3, project(M3, q3, 100 .* Z3))
        @test is_vector(M3, p3, log(M3, p3, q3))
    end

    Manifolds.Test.test_manifold(
        SPDFixedDeterminant(2, 1.0; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, manifold_dimension, repr], :Points => [p]),
        # Expectations
        Dict(
            get_embedding => SymmetricPositiveDefinite(2; parameter = :field),
            manifold_dimension => 2,
            repr => "SPDFixedDeterminant(2, 1.0; parameter=:field)",
        ),
    )

    # for n = 1 the manifold consists of the single point d
    Manifolds.Test.test_manifold(
        SPDFixedDeterminant(1, 2.0),
        Dict(
            :Functions => [
                exp, injectivity_radius, is_flat, is_point, manifold_dimension,
                parallel_transport_to, repr, zero_vector,
            ],
            :InvalidPoints => [fill(1.0, 1, 1)],
            :Points => [fill(2.0, 1, 1), fill(2.0, 1, 1)],
            :RetractionMethods => [ExponentialRetraction()],
            :Vectors => [zeros(1, 1)],
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            exp => fill(2.0, 1, 1),
            injectivity_radius => 0.0,
            (injectivity_radius, ExponentialRetraction()) => 0.0,
            is_flat => true,
            manifold_dimension => 0,
            parallel_transport_to => zeros(1, 1),
            repr => "SPDFixedDeterminant(1, 2.0)",
            :IsPointErrors => [DomainError],
        ),
    )
end
