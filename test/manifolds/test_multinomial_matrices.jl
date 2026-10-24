using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The manifold of multinomial matrices" begin
    @test_throws DomainError MultinomialMatrices(0, 2)

    M = MultinomialMatrices(3, 2)
    # columns are points on the probability simplex
    p1 = [0.5 0.5; 0.4 0.4; 0.1 0.1]
    p2 = [0.6 0.4; 0.3 0.5; 0.1 0.1]
    p3 = [0.3 0.6; 0.6 0.3; 0.1 0.1]
    X = [0.1 0.0; -0.1 0.05; 0.0 -0.05]
    X2 = [0.0 0.1; 0.05 -0.1; -0.05 0.0]
    Y = [1.0 -1.0; 0.5 0.0; -1.0 2.0] # an ambient vector at p1
    # the distance on the simplex is 2 arccos(∑ᵢ √(pᵢqᵢ)), combined over the columns
    d12 = sqrt(sum((2 * acos(sum(sqrt.(p1[:, j] .* p2[:, j]))))^2 for j in 1:2))
    # Invalids
    pf1 = [2.0 2.0; 0.0 0.0; 0.0 0.0] # columns do not sum to 1
    pf2 = [2.0, 0.0, 0.0] # wrong size
    Xf1 = [1.0 0.0; 0.0 0.0; 0.0 0.0] # first column does not sum to 0
    Xf2 = [-1.0, 0.0, 0.0] # wrong size

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Covectors => [Manifolds.RieszRepresenterCotangentVector(M, p1, X)],
            :EmbeddedPoints => [p1],
            :EmbeddedVectors => [Y],
            :Functions => [
                copy, copyto!, distance, embed, exp, flat, geodesic, get_embedding,
                injectivity_radius, inner, is_flat, is_point, is_vector, log,
                manifold_dimension, mid_point, norm, project, rand, repr,
                representation_size, sharp, shortest_geodesic, zero_vector,
            ],
            :InvalidPoints => AbstractArray[pf1, pf2],
            :InvalidVectors => AbstractArray[Xf1, Xf2],
            :Points => [p1, p2, p3],
            :SecondVector => X2,
            :Vectors => [X, project(M, p2, X2), project(M, p3, X)],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
            distance => d12,
            (embed, :Point) => p1,
            (embed, :Vector) => X,
            get_embedding => Euclidean(3, 2; parameter = :field),
            injectivity_radius => 0.0,
            # the Fisher–Rao metric ∑ XY/p
            inner => sum(X .* X2 ./ p1),
            is_flat => false,
            manifold_dimension => 4,
            norm => sqrt(sum(X .^ 2 ./ p1)),
            (project, :Point) => p1,
            (project, :Vector) => [0.75 -1.5; 0.3 -0.4; -1.05 1.9],
            repr => "MultinomialMatrices(3, 2)",
            representation_size => (3, 2),
            sharp => X,
            :IsPointErrors => [CompositeManifoldError, DomainError],
            :IsVectorErrors => [ComponentManifoldError{Int64, DomainError}, DomainError],
            :IsVectorBasepointError => CompositeManifoldError,
        ),
    )

    @testset "Simplex powers" begin
        @test ProbabilitySimplex(2)^3 === MultinomialMatrices(3, 3)
        @test ProbabilitySimplex(2)^(3,) === PowerManifold(ProbabilitySimplex(2), 3)
        @test ^(ProbabilitySimplex(2), 2) === MultinomialMatrices(3, 2)
        @test typeof(^(ProbabilitySimplex(2), 2)) == MultinomialMatrices{
            ManifoldsBase.TypeParameter{Tuple{3, 2}},
            ProbabilitySimplex{ManifoldsBase.TypeParameter{Tuple{2}}, :open},
        }
        @test Manifolds.get_iterator(ProbabilitySimplex(3)^4) === Base.OneTo(4)
    end

    @testset "Riemannian Gradient" begin
        G = [1.0 -1.0; 0.0 0.0; -1.0 1.0]
        Z = riemannian_gradient(M, p1, G)
        @test isapprox(M, p1, Z, [0.3 -0.3; -0.16 0.16; -0.14 0.14])
        # the Riemannian gradient satisfies ⟨X, V⟩_p = ⟨Y, V⟩ for tangent vectors V
        V = [1.0 0.0; -1.0 1.0; 0.0 -1.0]
        @test inner(M, p1, Z, V) ≈ dot(G, V)
    end

    @testset "Tangent space projection" begin
        Z = project(M, p1, Y)
        # the projection is orthogonal with respect to the Fisher–Rao metric
        V = [1.0 0.0; -1.0 1.0; 0.0 -1.0]
        @test isapprox(inner(M, p1, Y - Z, V), 0.0; atol = 1.0e-15)
        @test project(M, p1, Z) ≈ Z
    end

    @testset "field parameter" begin
        @test typeof(^(ProbabilitySimplex(2; parameter = :field), 2)) ==
            MultinomialMatrices{Tuple{Int64, Int64}, ProbabilitySimplex{Tuple{Int64}, :open}}
        Mf = MultinomialMatrices(3, 2; parameter = :field)
        @test repr(Mf) == "MultinomialMatrices(3, 2; parameter=:field)"
        @test get_embedding(Mf) == Euclidean(3, 2; parameter = :field)
    end
end
