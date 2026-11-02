using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The manifold of multinomial doubly stochastic matrices" begin
    @test_throws DomainError MultinomialDoubleStochastic(0)

    M = MultinomialDoubleStochastic(3)
    p1 = ones(3, 3) ./ 3
    p2 = [0.1 0.2 0.7; 0.2 0.7 0.1; 0.7 0.1 0.2]
    p3 = [0.1 0.4 0.5; 0.4 0.5 0.1; 0.5 0.1 0.4]
    X = [-0.1 0.0 0.1; 0.0 0.2 -0.2; 0.1 -0.2 0.1]
    X2 = [0.1 -0.1 0.0; -0.1 0.1 0.0; 0.0 0.0 0.0]
    Y = [0.0 0.1 -0.1; 0.1 -0.2 0.1; -0.1 0.1 0.0]
    # Invalids
    pf1 = [0.1 0.9 0.1; 0.1 0.9 0.1; 0.1 0.1 0.9] # columns do not sum to 1
    pf2 = [0.1 0.9 0.1; 0.8 0.05 0.15; 0.1 0.05 0.75] # sum(pf2, dims=2)[3] is 0.9
    pf3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # contains nonpositive entries
    Xf1 = [-0.1 0.0 0.1; -0.2 0.1 0.1; 0.2 -0.1 -0.1] # nonzero column sums
    Manifolds.Test.test_manifold(
        M,
        Dict(
            :EmbeddedPoints => [[1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]],
            :EmbeddedVectors => [[1.0 -1.0 0.0; 0.0 0.0 1.0; -1.0 1.0 1.1]],
            :Functions => [
                copy, copyto!, default_retraction_method, default_vector_transport_method,
                embed, get_embedding, inner, is_flat, is_point, is_vector, manifold_dimension,
                norm, project, rand, repr, representation_size, retract, vector_transport_to,
                zero_vector,
            ],
            :InvalidPoints => [pf1, pf2, pf2', pf3],
            :InvalidVectors => [Xf1, Xf1'],
            :Points => [p1, p2, p3],
            :RetractionMethods => [ProjectionRetraction()],
            :SecondVector => X2,
            :Vectors => [X, project(M, p2, Y), project(M, p3, Y)],
            :VectorTransportMethods => [ProjectionTransport()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-12,
            default_retraction_method => ProjectionRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            (embed, :Point) => p1,
            (embed, :Vector) => X,
            get_embedding => MultinomialMatrices(3, 3),
            # the metric at p1 is 3 times the Euclidean one
            inner => 3 * dot(X, X2),
            is_flat => false,
            manifold_dimension => 4,
            norm => sqrt(3 * dot(X, X)),
            repr => "MultinomialDoubleStochastic(3)",
            representation_size => (3, 3),
            (vector_transport_to, ProjectionTransport()) => project(M, p2, X),
            :IsPointErrors => [
                ManifoldDomainError, CompositeManifoldError, ManifoldDomainError,
                ManifoldDomainError,
            ],
            :IsVectorErrors => [ManifoldDomainError, CompositeManifoldError],
        ),
    )

    @testset "Basics" begin
        @test inner(M, p1, [1.0 1.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 1.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]) == 9.0
        @test_throws DomainError project(M, -ones(3, 3))
        @test project(M, p1) == p1
        Z = [1 2 3; 4 5 6; 7 8 10] # integer entries
        @test project!(M, zeros(3, 3), Z) ≈ project(M, float.(Z))
    end

    @testset "Riemannian Gradient" begin
        G = [1.0 -1.0 0.0; 0.0 0.0 1.0; -1.0 1.0 1.1]
        @test isapprox(M, p1, project(M, p1, p1 .* G), riemannian_gradient(M, p1, G))
    end

    @testset "Tangent space projection at the uniform point" begin
        M4 = MultinomialDoubleStochastic(4)
        Y4 = [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        X4 = [9.0 -3.0 -3.0 -3.0; -3.0 1.0 1.0 1.0; -3.0 1.0 1.0 1.0; -3.0 1.0 1.0 1.0] ./ 16
        @test project(M4, ones(4, 4) ./ 4, Y4) ≈ X4
        M2 = MultinomialDoubleStochastic(2)
        @test project(M2, ones(2, 2) ./ 2, [1.0 0.0; 0.0 0.0]) ≈ [0.25 -0.25; -0.25 0.25]
    end

    @testset "Sinkhorn's algorithm" begin
        # needs more than a hundred iterations
        X4 = [-2.0 0.0 2.0; 0.0 4.0 -4.0; 2.0 -4.0 2.0]
        @test is_point(M, retract(M, p3, X4, ProjectionRetraction()))
        # stalls with a column sum at the next number above one
        A = [0.1 0.5 0.6; 0.1 0.9 0.5; 0.3 0.7 0.9]
        @test_logs project(M, A)
        @test_logs (:warn,) project(M, A; maxiter = 1)
        @test_logs project(M, A; maxiter = 1, warn_nonconvergence = false)
    end

    @testset "default retraction of the doubly stochastic multinomial manifolds" begin
        p = [0.5 0.3 0.2; 0.3 0.4 0.3; 0.2 0.3 0.5]
        Z = [0.1 -0.05 -0.05; -0.05 0.1 -0.05; -0.05 -0.05 0.1]
        for N in (MultinomialDoubleStochastic(3), MultinomialSymmetric(3))
            @test default_retraction_method(N) === ProjectionRetraction()
            @test retract(N, p, Z) == retract(N, p, Z, ProjectionRetraction())
            @test is_point(N, retract(N, p, Z))
        end
        # the fused retraction scales the tangent vector
        q = ManifoldsBase.retract_fused(M, p, Z, 0.5, ProjectionRetraction())
        @test is_point(M, q)
        @test isapprox(M, q, retract(M, p, 0.5 .* Z, ProjectionRetraction()))
    end

    @testset "field parameter" begin
        Mf = MultinomialDoubleStochastic(3; parameter = :field)
        @test repr(Mf) == "MultinomialDoubleStochastic(3; parameter=:field)"
        @test get_embedding(Mf) === MultinomialMatrices(3, 3; parameter = :field)
    end
end
