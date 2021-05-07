include("utils.jl")

@testset "Probability simplex" begin
    M = ProbabilitySimplex(2)
    @test M^4 == MultinomialMatrices(3, 4)
    p = [0.1, 0.7, 0.2]
    q = [0.3, 0.6, 0.1]
    X = zeros(3)
    Y = [-0.1, 0.05, 0.05]
    @test is_point(M, p)
    @test_throws DomainError is_point(M, p .+ 1, true)
    @test_throws DomainError is_point(M, [0], true)
    @test_throws DomainError is_point(M, -ones(3), true)
    @test manifold_dimension(M) == 2
    @test is_tangent_vector(M, p, X)
    @test is_tangent_vector(M, p, Y)
    @test_throws DomainError is_tangent_vector(M, p .+ 1, X, true)
    @test_throws DomainError is_tangent_vector(M, p, zeros(4), true)
    @test_throws DomainError is_tangent_vector(M, p, Y .+ 1, true)

    @test Manifolds.default_metric_dispatch(M, Manifolds.FisherRaoMetric()) === Val{true}()

    @test injectivity_radius(M, p) == injectivity_radius(M, p, ExponentialRetraction())
    @test injectivity_radius(M, p, SoftmaxRetraction()) == injectivity_radius(M, p)
    @test injectivity_radius(M, ExponentialRetraction()) == 0
    @test injectivity_radius(M) == 0
    @test injectivity_radius(M, SoftmaxRetraction()) == 0

    pE = similar(p)
    embed!(M, pE, p)
    @test pE == p
    XE = similar(X)
    embed!(M, XE, p, X)
    @test XE == X

    @test mean(M, [p, q]) == shortest_geodesic(M, p, q)(0.5)

    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        @testset "Type $T" begin
            pts = [
                convert(T, [0.5, 0.3, 0.2]),
                convert(T, [0.4, 0.4, 0.2]),
                convert(T, [0.3, 0.5, 0.2]),
            ]
            test_manifold(
                M,
                pts,
                test_injectivity_radius=false,
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_vee_hat=false,
                test_forward_diff=false,
                test_reverse_diff=false,
                is_tangent_atol_multiplier=5.0,
                inverse_retraction_methods=[SoftmaxInverseRetraction()],
                retraction_methods=[SoftmaxRetraction()],
            )
        end
    end

    @testset "Projection testing" begin
        p = [1 / 3, 1 / 3, 1 / 3]
        q = [0.2, 0.3, 0.5]
        X = log(M, p, q)
        X2 = X .+ 10
        Y = project(M, p, X2)
        @test isapprox(M, p, X, Y)

        X = log(M, q, p)
        X2 = X + [1, 2, 3]
        Y = project(M, q, X2)
        @test is_tangent_vector(M, q, Y; atol=1e-15)

        @test_throws DomainError project(M, [1, -1, 2])
        @test isapprox(M, [0.6, 0.2, 0.2], project(M, [0.3, 0.1, 0.1]))
    end
end
