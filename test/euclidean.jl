include("utils.jl")

@testset "Euclidean" begin
    E = Euclidean(3)
    Ec = Euclidean(3; field = ℂ)
    EM = Manifolds.MetricManifold(E, Manifolds.EuclideanMetric())
    @test repr(E) == "Euclidean(3; field = ℝ)"
    @test repr(Ec) == "Euclidean(3; field = ℂ)"
    @test repr(Euclidean(2, 3; field = ℍ)) == "Euclidean(2, 3; field = ℍ)"
    @test Manifolds.allocation_promotion_function(Ec, get_vector, ()) === complex
    @test is_default_metric(EM)
    @test is_default_metric(E, Manifolds.EuclideanMetric())
    @test Manifolds.default_metric_dispatch(E, Manifolds.EuclideanMetric()) === Val{true}()
    p = zeros(3)
    @test det_local_metric(EM, p) == one(eltype(p))
    @test log_local_metric_density(EM, p) == zero(eltype(p))
    @test project!(E, p, p) == p
    @test embed!(E, p, p) == p
    @test manifold_dimension(Ec) == 2 * manifold_dimension(E)
    X = zeros(3)
    X[1] = 1.0
    Y = similar(X)
    project!(E, Y, p, X)
    @test Y == X

    # real manifold does not allow complex values
    @test_throws DomainError is_manifold_point(Ec, [:a, :b, :b], true)
    @test_throws DomainError is_manifold_point(E, [1.0, 1.0im, 0.0], true)
    @test_throws DomainError is_manifold_point(E, [1], true)
    @test_throws DomainError is_tangent_vector(Ec, [:a, :b, :b], [1.0, 1.0, 0.0], true)
    @test_throws DomainError is_tangent_vector(E, [1.0, 1.0im, 0.0], [1.0, 1.0, 0.0], true) # real manifold does not allow complex values
    @test_throws DomainError is_tangent_vector(E, [1], [1.0, 1.0, 0.0], true)
    @test_throws DomainError is_tangent_vector(E, [0.0, 0.0, 0.0], [1.0], true)
    @test_throws DomainError is_tangent_vector(E, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0im], true)
    @test_throws DomainError is_tangent_vector(Ec, [0.0, 0.0, 0.0], [:a, :b, :c], true)

    @test E^2 === Euclidean(3, 2)
    @test ^(E, 2) === Euclidean(3, 2)
    @test E^(2,) === Euclidean(3, 2)
    @test Ec^(4, 5) === Euclidean(3, 4, 5; field = ℂ)

    manifolds = [E, EM, Ec]
    types = [Vector{Float64}]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    TEST_DOUBLE64 && push!(types, Vector{Double64})
    TEST_STATIC_SIZED && push!(types, MVector{3,Float64})

    types_complex = [Vector{ComplexF64}]
    TEST_FLOAT32 && push!(types_complex, Vector{ComplexF32})
    TEST_DOUBLE64 && push!(types_complex, Vector{ComplexDF64})
    TEST_STATIC_SIZED && push!(types_complex, MVector{3,ComplexF64})

    for M in manifolds
        basis_types = if M == E
            (
                DefaultOrthonormalBasis(),
                ProjectedOrthonormalBasis(:svd),
                DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0]),
            )
        elseif M == Ec
            (
                DefaultOrthonormalBasis(),
                DefaultOrthonormalBasis(ℂ),
                DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0]),
            )
        else
            ()
        end
        for T in types
            @testset "$M Type $T" begin
                pts = [
                    convert(T, [1.0, 0.0, 0.0]),
                    convert(T, [0.0, 1.0, 0.0]),
                    convert(T, [0.0, 0.0, 1.0]),
                ]
                test_manifold(
                    M,
                    pts,
                    test_reverse_diff = isa(T, Vector),
                    test_project_point = true,
                    test_project_tangent = true,
                    test_musical_isomorphisms = true,
                    test_default_vector_transport = true,
                    vector_transport_methods = [
                        ParallelTransport(),
                        SchildsLadderTransport(),
                        PoleLadderTransport(),
                    ],
                    test_mutating_rand = isa(T, Vector),
                    point_distributions = [Manifolds.projected_distribution(
                        M,
                        Distributions.MvNormal(zero(pts[1]), 1.0),
                    )],
                    tvector_distributions = [Manifolds.normal_tvector_distribution(
                        M,
                        pts[1],
                        1.0,
                    )],
                    basis_types_vecs = basis_types,
                    basis_types_to_from = basis_types,
                    basis_has_specialized_diagonalizing_get = true,
                    test_vee_hat = isa(M, Euclidean),
                )
            end
        end
    end
    for T in types_complex
        @testset "Complex Euclidean, type $T" begin
            pts = [
                convert(T, [1.0im, -1.0im, 1.0]),
                convert(T, [0.0, 1.0, 1.0im]),
                convert(T, [0.0, 0.0, 1.0]),
            ]
            test_manifold(
                Ec,
                pts,
                test_reverse_diff = isa(T, Vector),
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                test_vee_hat = false,
            )
        end
    end

    @testset "hat/vee" begin
        E = Euclidean(3, 2)
        p = collect(reshape(1.0:6.0, (3, 2)))
        X = collect(reshape(7.0:12.0, (3, 2)))
        @test hat(E, p, vec(X)) ≈ X
        Y = allocate(X)
        @test hat!(E, Y, p, vec(X)) === Y
        @test Y ≈ X
        @test vee(E, p, X) ≈ vec(X)
        Y = allocate(vec(X))
        @test vee!(E, Y, p, X) === Y
        @test Y ≈ vec(X)
    end

    @testset "Number systems power" begin
        @test ℝ^2 === Euclidean(2)
        @test ℝ^(2, 3) === Euclidean(2, 3)

        @test ℂ^2 === Euclidean(2; field = ℂ)
        @test ℂ^(2, 3) === Euclidean(2, 3; field = ℂ)

        @test ℍ^2 === Euclidean(2; field = ℍ)
        @test ℍ^(2, 3) === Euclidean(2, 3; field = ℍ)
    end

    @testset "Embeddings into larger Euclidean Manifolds" begin
        M = Euclidean(3,3)
        N = Euclidean(4,4)
        O = EmbeddedManifold(M,N)
        # first test with same length of sizes
        p = ones(3,3)
        q = zeros(4,4)
        qT = zeros(4,4)
        qT[1:3,1:3] .= 1.
        embed!(O,q,p)
        @test norm(qT-q) == 0
        # test with different sizes, check that it only fills first element
        q2 = zeros(4,4,3)
        q2T = zeros(4,4,3)
        q2T[1:3,1:3,1] .= 1.
        embed!(O,q2,p)
        @test norm(q2T-q2) == 0
        # wrong size error checks
        @test_throws DomainError embed!(O,zeros(3,3),zeros(3,3,5))
        @test_throws DomainError embed!(O,zeros(3,3),zeros(4,4))
    end
end
