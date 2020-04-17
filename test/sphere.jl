include("utils.jl")

@testset "Sphere" begin
    M = Sphere(2)
    @testset "Sphere Basics" begin
        @test repr(M) == "Sphere(2, ℝ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3},ℝ}
        @test representation_size(M) == (3,)
        @test injectivity_radius(M) == π
        @test injectivity_radius(M, ExponentialRetraction()) == π
        @test injectivity_radius(M, ProjectionRetraction()) == π / 2
        @test base_manifold(M) === M
        @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
        @test !is_tangent_vector(M, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])
        @test_throws DomainError is_manifold_point(M, [2.0, 0.0, 0.0], true)
        @test !is_manifold_point(M, [2.0, 0.0, 0.0])
        @test !is_tangent_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws DomainError is_tangent_vector(
            M,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            true,
        )
        @test injectivity_radius(M, [1.0, 0.0, 0.0], ProjectionRetraction()) == π / 2
    end
    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    for T in types
        @testset "Type $T" begin
            pts = [
                convert(T, [1.0, 0.0, 0.0]),
                convert(T, [0.0, 1.0, 0.0]),
                convert(T, [0.0, 0.0, 1.0]),
            ]
            test_manifold(
                M,
                pts,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_vector_transport = true,
                test_mutating_rand = isa(T, Vector),
                point_distributions = [Manifolds.uniform_distribution(M, pts[1])],
                tvector_distributions = [Manifolds.normal_tvector_distribution(
                    M,
                    pts[1],
                    1.0,
                )],
                basis_types_vecs = (DiagonalizingOrthonormalBasis([0.0, 1.0, 2.0]),),
                basis_types_to_from = basis_types,
                test_vee_hat = false,
                retraction_methods = [ProjectionRetraction(), ExponentialRetraction()],
                inverse_retraction_methods = [ProjectionInverseRetraction()],
            )
            @test isapprox(-pts[1], exp(M, pts[1], log(M, pts[1], -pts[1])))
        end
    end

    @testset "Distribution tests" begin
        usd_mvector = Manifolds.uniform_distribution(M, @MVector [1.0, 0.0, 0.0])
        @test isa(rand(usd_mvector), MVector)

        gtsd_mvector =
            Manifolds.normal_tvector_distribution(M, (@MVector [1.0, 0.0, 0.0]), 1.0)
        @test isa(rand(gtsd_mvector), MVector)
    end

    @testset "Embedding test" begin
        p = [1.0, 0.0, 0.0]
        X = [0.0, 1.0, 1.0]
        @test embed(M, p) == p
        q = similar(p)
        embed!(M, q, p)
        @test q == p
        @test embed(M, p, X) == X
        Y = similar(X)
        embed!(M, Y, p, X)
        @test Y == X
    end

    @testset "log edge case" begin
        n = manifold_dimension(M)
        x = normalize(randn(n + 1))
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol = 1e-12)
        vexp = normalize(project(M, x, [1, zeros(n)...]))
        @test v ≈ π * vexp

        x = [1, zeros(n)...]
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol = 1e-12)
        vexp = normalize(project(M, x, [0, 1, zeros(n - 1)...]))
        @test v ≈ π * vexp
    end

    @testset "Complex Sphere" begin
        M = Sphere(2, ℂ)
        @test repr(M) == "Sphere(2, ℂ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3},ℂ}
        @test representation_size(M) == (3,)
        p = [1.0, 1.0im, 1.0]
        q = project(M, p)
        @test is_manifold_point(M, q)
        Y = [2.0, 1.0im, 20.0]
        X = project(M, q, Y)
        @test is_tangent_vector(M, q, X, true; atol = 10^(-14))
    end

    @testset "Tensor Sphere" begin
        M = ArraySphere(2, 2; field = ℝ)
        @test repr(M) == "ArraySphere(2, 2; field = ℝ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℝ}
        @test representation_size(M) == (2, 2)
        p = ones(2, 2)
        q = project(M, p)
        @test is_manifold_point(M, q)
        Y = [1.0 0.0; 0.0 1.1]
        X = project(M, q, Y)
        @test is_tangent_vector(M, q, X)
        M = ArraySphere(2, 2; field = ℂ)

        @test repr(M) == "ArraySphere(2, 2; field = ℂ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℂ}
        @test representation_size(M) == (2, 2)
    end
end
