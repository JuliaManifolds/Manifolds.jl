using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Multinomial symmetric matrices" begin
    @test_throws DomainError MultinomialSymmetric(0)

    M = MultinomialSymmetric(3)
    p1 = ones(3, 3) ./ 3
    p2 = [0.1 0.2 0.7; 0.2 0.7 0.1; 0.7 0.1 0.2]
    p3 = [0.1 0.4 0.5; 0.4 0.5 0.1; 0.5 0.1 0.4]
    # symmetric matrices with zero row sums are tangent vectors at every point
    X = [-0.1 0.0 0.1; 0.0 0.2 -0.2; 0.1 -0.2 0.1]
    X2 = [0.1 -0.1 0.0; -0.1 0.1 0.0; 0.0 0.0 0.0]
    Y = [0.0 0.1 -0.1; 0.1 -0.2 0.1; -0.1 0.1 0.0]
    # Invalids
    pf1 = [0.1 0.9 0.1; 0.1 0.9 0.1; 0.1 0.1 0.9] # not symmetric
    pf2 = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.9] # rows and columns do not sum to 1
    pf3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] # contains nonpositive entries
    Xf1 = [0.0 1.0 -1.0; 0.0 0.0 0.0; 0.0 0.0 0.0] # not symmetric
    Xf2 = [0.0 -1.0 0.0; -1.0 0.0 0.0; 0.0 0.0 0.0] # nonzero row sums
    E11 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :EmbeddedPoints => [2 .* p1],
            :EmbeddedVectors => [E11],
            :Functions => [
                copy, copyto!, default_retraction_method, default_vector_transport_method,
                embed, get_embedding, inner, is_flat, is_point, is_vector, manifold_dimension,
                norm, project, rand, repr, representation_size, retract, vector_transport_to,
                zero_vector,
            ],
            :InvalidPoints => [pf1, pf2, pf3],
            :InvalidVectors => [Xf1, Xf2],
            :Points => [p1, p2, p3],
            :RetractionMethods => [ProjectionRetraction()],
            :SecondVector => X2,
            :Vectors => [X, Y, X2],
            :VectorTransportMethods => [ProjectionTransport()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            default_retraction_method => ProjectionRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            (embed, :Point) => p1,
            (embed, :Vector) => X,
            get_embedding => MultinomialMatrices(3, 3),
            # the metric at p1 is 3 times the Euclidean one
            inner => 3 * dot(X, X2),
            is_flat => false,
            manifold_dimension => 3,
            norm => sqrt(3 * dot(X, X)),
            # Sinkhorn's algorithm scales 2p1 back to p1, and at p1 the projection of E₁₁
            # subtracts (αᵢ + αⱼ)/3 with α = (5, -1, -1)/6 from (I + p1)α = (1, 0, 0)
            (project, :Point) => p1,
            (project, :Vector) => [4.0 -2.0 -2.0; -2.0 1.0 1.0; -2.0 1.0 1.0] ./ 9,
            repr => "MultinomialSymmetric(3)",
            representation_size => (3, 3),
            (vector_transport_to, ProjectionTransport()) => X,
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
            :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError],
        ),
    )

    @testset "Basics" begin
        @test is_flat(MultinomialSymmetric(2))
        @test inner(M, p1, [1.0 1.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 1.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]) == 9.0
        @test_throws DomainError project(M, -ones(3, 3))
        @test project(M, p2) ≈ p2
        # tangent vectors with row sums 1.0e-6
        Xf3 = X .+ 1.0e-6 / 3
        @test !is_vector(M, p1, Xf3)
        @test is_vector(M, p1, Xf3; atol = 1.0e-3)
        # a doubly stochastic scaling of 4p2 is p2 again
        @test retract(M, p2, zero(X)) ≈ p2
        @test ManifoldsBase.retract_fused(M, p2, X, 0.0, ProjectionRetraction()) ≈ p2
        @test ManifoldsBase.retract_fused(M, p2, X, 0.5, ProjectionRetraction()) ≈
            retract(M, p2, 0.5 .* X, ProjectionRetraction())
    end

    @testset "Nonsymmetric input from the embedding" begin
        B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 10.0]
        q = project(M, B)
        @test is_point(M, q; error = :error)
        @test q == project(M, (B + B') / 2)
        Z = [1.0 2.0 0.0; 0.0 1.0 -1.0; 3.0 0.0 1.0]
        @test is_vector(M, p2, project(M, p2, Z))
        @test project(M, p2, Z) ≈ project(M, p2, (Z + Z') / 2)
        @test is_vector(M, p2, riemannian_gradient(M, p2, Z))
        H = [1.0 0.5 2.0; 0.3 1.0 0.0; 1.5 0.2 1.0]
        @test riemannian_Hessian(M, p2, Z, H, X) ≈
            riemannian_Hessian(M, p2, (Z + Z') / 2, (H + H') / 2, X)
    end

    @testset "Riemannian Hessian" begin
        G = zeros(3, 3)
        H = 0.5 * one(p1)
        Z = riemannian_Hessian(M, p1, G, H, X)
        Z2 = similar(Z)
        riemannian_Hessian!(M, Z2, p1, G, H, X)
        @test Z2 ≈ Z
        @test is_vector(M, p1, Z)
        # for a vanishing Euclidean gradient this is the Riemannian gradient conversion
        # p ⊙ H of the Euclidean Hessian H applied to X, projected onto the tangent space
        @test Z ≈ project(M, p1, p1 .* H)
    end

    @testset "random points with a tolerance for Sinkhorn's algorithm" begin
        A = rand(MersenneTwister(1), SymmetricMatrices(3))
        @test rand(MersenneTwister(1), M; tolerance = 0.1) == project(M, A; tolerance = 0.1)
    end

    @testset "field parameter" begin
        Mf = MultinomialSymmetric(3; parameter = :field)
        @test repr(Mf) == "MultinomialSymmetric(3; parameter=:field)"
        @test get_embedding(Mf) === MultinomialMatrices(3, 3; parameter = :field)
    end
end
