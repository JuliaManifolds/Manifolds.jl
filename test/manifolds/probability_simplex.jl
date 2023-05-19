include("../utils.jl")

@testset "Probability simplex" begin
    M = ProbabilitySimplex(2)
    @test M^4 == MultinomialMatrices(3, 4)
    p = [0.1, 0.7, 0.2]
    q = [0.3, 0.6, 0.1]
    X = zeros(3)
    Y = [-0.1, 0.05, 0.05]
    @test is_point(M, p)
    @test_throws DomainError is_point(M, p .+ 1, true)
    @test_throws ManifoldDomainError is_point(M, [0], true)
    @test_throws DomainError is_point(M, -ones(3), true)
    @test manifold_dimension(M) == 2
    @test !is_flat(M)
    @test is_vector(M, p, X)
    @test is_vector(M, p, Y)
    @test_throws ManifoldDomainError is_vector(M, p .+ 1, X, true)
    @test_throws ManifoldDomainError is_vector(M, p, zeros(4), true)
    @test_throws DomainError is_vector(M, p, Y .+ 1, true)

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
                is_tangent_atol_multiplier=5.0,
                inverse_retraction_methods=[SoftmaxInverseRetraction()],
                retraction_methods=[SoftmaxRetraction()],
                test_inplace=true,
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

        # Check adaption of metric and representer
        Y1 = change_metric(M, EuclideanMetric(), p, X)
        @test Y1 == X .* sqrt.(p)
        Y2 = change_representer(M, EuclideanMetric(), p, X)
        @test Y2 == X .* p

        X = log(M, q, p)
        X2 = X + [1, 2, 3]
        Y = project(M, q, X2)
        @test is_vector(M, q, Y; atol=1e-15)

        @test_throws DomainError project(M, [1, -1, 2])
        @test isapprox(M, [0.6, 0.2, 0.2], project(M, [0.3, 0.1, 0.1]))
    end

    @testset "Gradient conversion" begin
        M = ProbabilitySimplex(4) # n=5
        # For f(p) = \sum_i 1/n log(p_i) we know the Euclidean gradient
        grad_f_eucl(p) = [1 / (5 * pi) for pi in p]
        # but we can also easily derive the Riemannian one
        grad_f(M, p) = 1 / 5 .- p
        #We take some point
        p = [0.1, 0.2, 0.4, 0.2, 0.1]
        Y = grad_f_eucl(p)
        X = grad_f(M, p)
        @test isapprox(M, p, X, riemannian_gradient(M, p, Y))
        Z = zero_vector(M, p)
        riemannian_gradient!(M, Z, p, Y)
        @test X == Z
    end

    @testset "Simplex with boundary" begin
        Mb = ProbabilitySimplex(2; closed=:Closed)
        p = [0, 0.5, 0.5]
        X = [0, 1, -1]
        Y = [0, 2, -2]
        @test is_point(Mb, p)
        @test inner(Mb, p, X, Y) == 8

        @test_throws ArgumentError ProbabilitySimplex(2; closed=:tomato)
    end
end
