using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Multinomial symmetric positive definite matrices" begin
    @test_throws DomainError MultinomialSymmetricPositiveDefinite(0)

    M = MultinomialSymmetricPositiveDefinite(3)
    p1 = [0.6 0.2 0.2; 0.2 0.6 0.2; 0.2 0.2 0.6]
    p2 = [0.5 0.3 0.2; 0.3 0.4 0.3; 0.2 0.3 0.5]
    p3 = [0.4 0.3 0.3; 0.3 0.4 0.3; 0.3 0.3 0.4]
    # tangent vectors are symmetric with zero row sums
    X = [0.2 -0.1 -0.1; -0.1 0.2 -0.1; -0.1 -0.1 0.2]
    X2 = [0.0 0.1 -0.1; 0.1 -0.2 0.1; -0.1 0.1 0.0]
    Y = [0.1 0.2 -0.3; 0.2 0.1 0.05; -0.3 0.05 0.1]
    # (α1ᵀ + 1αᵀ) ⊙ p and skew-symmetric matrices are normal to the tangent space at p
    α = [1.0, 2.0, 3.0]
    N1 = (α .+ α') .* p1
    S = [0.0 1.0 -2.0; -1.0 0.0 0.5; 2.0 -0.5 0.0]
    # Invalids
    pf1 = zeros(3, 3) # symmetric but does not sum to 1
    pf2 = [0.3 0.4 0.3; 0.4 0.2 0.4; 0.3 0.4 0.3] # multinomial, eigenvalues -0.2, 0.0 and 1.0
    pf3 = [0.2 0.3 0.5; 0.4 0.2 0.4; 0.4 0.5 0.1] # multinomial but not symmetric
    Xf1 = ones(3, 3) # symmetric but does not sum to zero
    Xf2 = [-0.5 0.3 0.2; 0.2 -0.5 0.3; 0.3 0.2 -0.5] # sums to zero but not symmetric

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :EmbeddedPoints => [2 .* p1],
            :EmbeddedVectors => [X .+ N1 .+ S],
            :Functions => [
                copy, copyto!, default_retraction_method, embed, get_embedding, inner,
                is_flat, is_point, is_vector, manifold_dimension, norm, project, rand, repr,
                representation_size, retract, zero_vector,
            ],
            :InvalidPoints => [pf1, pf2, pf3],
            :InvalidVectors => [Xf1, Xf2],
            :Points => [p1, p2, p3],
            :RetractionMethods => [ProjectionRetraction()],
            :SecondVector => X2,
            :Vectors => [X, project(M, p2, Y), project(M, p3, Y)],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-12,
            default_retraction_method => ProjectionRetraction(),
            (embed, :Point) => p1,
            (embed, :Vector) => X,
            get_embedding => MultinomialSymmetric(3),
            # the Fisher metric ∑ XY/p of the embedding
            inner => sum(X .* X2 ./ p1),
            is_flat => false,
            manifold_dimension => 3,
            norm => sqrt(sum(X .^ 2 ./ p1)),
            (project, :Point) => p1,
            (project, :Vector) => X,
            repr => "MultinomialSymmetricPositiveDefinite(3)",
            representation_size => (3, 3),
            :IsPointErrors => [ManifoldDomainError, ManifoldDomainError, ManifoldDomainError],
            :IsVectorErrors => [ManifoldDomainError, ManifoldDomainError],
            :IsVectorBasepointError => ManifoldDomainError,
        ),
    )

    @testset "Basics" begin
        @test is_flat(MultinomialSymmetricPositiveDefinite(2))
        @test manifold_dimension(MultinomialSymmetricPositiveDefinite(2)) == 1
    end

    @testset "Random" begin
        @test is_point(M, rand!(MersenneTwister(), M, zeros(3, 3)); error = :error, atol = 5.0e-14)
        @test is_point(M, rand(MersenneTwister(31), M); error = :error)
        @test is_vector(M, p1, rand(MersenneTwister(31), M; vector_at = p1); error = :error)
    end

    @testset "Fisher metric, tangent projection and retraction of the embedding" begin
        N = MultinomialSymmetric(3)
        G = [1.0 2.0 3.0; 0.5 -1.0 2.0; 1.5 0.0 -2.0]
        Z = project(M, p2, Y)
        @test Z ≈ project(N, p2, Y)
        @test inner(M, p2, Z, Z) ≈ inner(N, p2, Z, Z)
        @test riemannian_gradient(M, p2, G) ≈ riemannian_gradient(N, p2, G)
        q = retract(M, p2, 0.1 * Z, ProjectionRetraction())
        @test q ≈ retract(N, p2, 0.1 * Z, ProjectionRetraction())
        @test ManifoldsBase.retract_fused(M, p2, Z, 0.1, ProjectionRetraction()) ≈ q
        @test isposdef(Symmetric(q))
    end

    Manifolds.Test.test_manifold(
        MultinomialSymmetricPositiveDefinite(3; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, manifold_dimension, repr], :Points => [p1]),
        # Expectations
        Dict(
            get_embedding => MultinomialSymmetric(3; parameter = :field),
            manifold_dimension => 3,
            repr => "MultinomialSymmetricPositiveDefinite(3; parameter=:field)",
        ),
    )
end
