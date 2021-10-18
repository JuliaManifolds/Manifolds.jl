include("../utils.jl")

using Manifolds: induced_basis
using ManifoldsBase: TFVector

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
        @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
        @test !is_vector(M, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])
        @test_throws DomainError is_point(M, [2.0, 0.0, 0.0], true)
        @test !is_point(M, [2.0, 0.0, 0.0])
        @test !is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws DomainError is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], true)
        @test injectivity_radius(M, [1.0, 0.0, 0.0], ProjectionRetraction()) == π / 2
    end
    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    basis_types = (DefaultOrthonormalBasis(), ProjectedOrthonormalBasis(:svd))
    test_atlases = (Manifolds.StereographicAtlas(), Manifolds.RetractionAtlas())
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
                test_reverse_diff=isa(T, Vector),
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                test_mutating_rand=isa(T, Vector),
                point_distributions=[Manifolds.uniform_distribution(M, pts[1])],
                tvector_distributions=[
                    Manifolds.normal_tvector_distribution(M, pts[1], 1.0),
                ],
                basis_types_vecs=(DiagonalizingOrthonormalBasis([0.0, 1.0, 2.0]),),
                basis_types_to_from=basis_types,
                test_vee_hat=false,
                retraction_methods=[ProjectionRetraction(), ExponentialRetraction()],
                inverse_retraction_methods=[ProjectionInverseRetraction()],
                is_tangent_atol_multiplier=1,
                test_atlases=test_atlases,
                test_inplace=true,
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

    @testset "DefaultOrthonormalBasis edge cases" begin
        B = DefaultOrthonormalBasis(ℝ)
        n = manifold_dimension(M)
        p = [1; zeros(n)]
        for i in 1:n  # p'x ± 1
            vcoord = [j == i for j in 1:n]
            v = [0; vcoord]
            @test get_coordinates(M, p, v, B) ≈ vcoord
            @test get_vector(M, p, vcoord, B) ≈ v
            @test get_coordinates(M, -p, v, B) ≈ vcoord
            @test get_vector(M, -p, vcoord, B) ≈ v
        end
        p = [0; 1; zeros(n - 1)] # p'x = 0
        for i in 1:n
            vcoord = [j == i for j in 1:n]
            v = get_vector(M, p, vcoord, B)
            @test is_vector(M, p, v)
            @test get_coordinates(M, p, v, B) ≈ vcoord
        end
    end

    @testset "log edge case" begin
        n = manifold_dimension(M)
        x = normalize(randn(n + 1))
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol=1e-12)
        vexp = normalize(project(M, x, [1, zeros(n)...]))
        @test v ≈ π * vexp

        x = [1, zeros(n)...]
        v = log(M, x, -x)
        @test norm(v) ≈ π
        @test isapprox(dot(x, v), 0; atol=1e-12)
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
        @test is_point(M, q)
        Y = [2.0, 1.0im, 20.0]
        X = project(M, q, Y)
        @test is_vector(M, q, X, true; atol=10^(-14))
    end

    @testset "Quaternion Sphere" begin
        M = Sphere(2, ℍ)
        @test repr(M) == "Sphere(2, ℍ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{3},ℍ}
        @test representation_size(M) == (3,)
        p = [Quaternion(1.0), Quaternion(1.0im), Quaternion(0.0, 0.0, -1.0, 0.0)]
        q = project(M, p)
        @test is_point(M, q)
        Y = [Quaternion(2.0), Quaternion(1.0im), Quaternion(0.0, 0.0, 20.0, 0.0)]
        X = project(M, q, Y)
        @test is_vector(M, q, X, true; atol=10^(-14))
    end

    @testset "Array Sphere" begin
        M = ArraySphere(2, 2; field=ℝ)
        @test repr(M) == "ArraySphere(2, 2; field = ℝ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℝ}
        @test representation_size(M) == (2, 2)
        p = ones(2, 2)
        q = project(M, p)
        @test is_point(M, q)
        Y = [1.0 0.0; 0.0 1.1]
        X = project(M, q, Y)
        @test is_vector(M, q, X)
        M = ArraySphere(2, 2; field=ℂ)

        @test repr(M) == "ArraySphere(2, 2; field = ℂ)"
        @test typeof(get_embedding(M)) === Euclidean{Tuple{2,2},ℂ}
        @test representation_size(M) == (2, 2)
    end

    @testset "StereographicAtlas" begin
        M = Sphere(2)
        A = Manifolds.StereographicAtlas()
        p = randn(3)
        p ./= norm(p)
        for k in [1, -1]
            p *= k
            i = Manifolds.get_chart_index(M, A, p)
            @test i === (p[1] < 0 ? :south : :north)
            a = get_parameters(M, A, i, p)
            q = get_point(M, A, i, a)
            @test isapprox(M, p, q)

            p2 = randn(3)
            p2 ./= norm(p2)
            p3 = randn(3)
            p3 ./= norm(p3)
            X2 = log(M, p, p2)
            X3 = log(M, p, p3)
            B = induced_basis(M, A, i, TangentSpace)

            X2B = get_coordinates(M, p, X2, B)
            X3B = get_coordinates(M, p, X3, B)

            @test inner(M, p, X2, X3) ≈ dot(X2B, local_metric(M, p, B) * X3B)

            X2back = get_vector(M, p, X2B, B)
            X3back = get_vector(M, p, X3B, B)
            @test isapprox(M, p, X2, X2back)
            @test isapprox(M, p, X3, X3back)

            @testset "transition_map" begin
                other_chart = i === :south ? :north : :south
                a_other = Manifolds.transition_map(M, A, i, other_chart, a)
                @test isapprox(M, p, get_point(M, A, other_chart, a_other))

                a_other2 = allocate(a_other)
                Manifolds.transition_map!(M, a_other2, A, i, A, other_chart, a)
                @test isapprox(M, p, get_point(M, A, other_chart, a_other2))

                a_other2 = allocate(a_other)
                Manifolds.transition_map!(M, a_other2, A, i, other_chart, a)
                @test isapprox(M, p, get_point(M, A, other_chart, a_other2))
            end
        end
    end

    @testset "Metric conversion is the identity" begin
        p = [1.0, 0.0, 0.0]
        X = [0.0, 1.0, 1.0]
        Y = change_representer(M, EuclideanMetric(), p, X)
        @test Y == X
        Z = change_metric(M, EuclideanMetric(), p, X)
        @test Z == X
        @test local_metric(M, p, DefaultOrthonormalBasis()) == Diagonal([1.0, 1.0])
    end
end
