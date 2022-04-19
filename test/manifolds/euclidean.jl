include("../utils.jl")

using Manifolds: induced_basis

@testset "Euclidean" begin
    E = Euclidean(3)
    Ec = Euclidean(3; field=ℂ)
    EM = Manifolds.MetricManifold(E, Manifolds.EuclideanMetric())
    @test repr(E) == "Euclidean(3; field = ℝ)"
    @test repr(Ec) == "Euclidean(3; field = ℂ)"
    @test repr(Euclidean(2, 3; field=ℍ)) == "Euclidean(2, 3; field = ℍ)"
    @test Manifolds.allocation_promotion_function(Ec, get_vector, ()) === complex
    p = zeros(3)
    A = Manifolds.RetractionAtlas()
    B = induced_basis(EM, A, p, TangentSpace)
    @test det_local_metric(EM, p, B) == one(eltype(p))
    @test log_local_metric_density(EM, p, B) == zero(eltype(p))
    @test project!(E, p, p) == p
    @test embed!(E, p, p) == p
    @test manifold_dimension(Ec) == 2 * manifold_dimension(E)
    X = zeros(3)
    X[1] = 1.0
    Y = similar(X)
    project!(E, Y, p, X)
    @test Y == X
    @test embed(E, p, X) == X

    # temp: explicit test for induced basis
    B = induced_basis(E, RetractionAtlas(), 0, ManifoldsBase.TangentSpaceType())
    @test get_coordinates(E, p, X, B) == X
    get_coordinates!(E, Y, p, X, B)
    @test Y == X
    @test get_vector(E, p, Y, B) == X
    Y2 = similar(X)
    get_vector!(E, Y2, p, Y, B)
    @test Y2 == X

    Y = parallel_transport_along(E, p, X, [p])
    @test Y == X
    parallel_transport_along!(E, Y, p, X, [p])
    @test Y == X

    Y = vector_transport_along(E, p, X, [p])
    @test Y == X
    vector_transport_along!(E, Y, p, X, [p])
    @test Y == X

    # real manifold does not allow complex values
    @test_throws DomainError is_point(Ec, [:a, :b, :b], true)
    @test_throws DomainError is_point(E, [1.0, 1.0im, 0.0], true)
    @test_throws DomainError is_point(E, [1], true)
    @test_throws DomainError is_vector(Ec, [:a, :b, :b], [1.0, 1.0, 0.0], true)
    @test_throws DomainError is_vector(E, [1.0, 1.0im, 0.0], [1.0, 1.0, 0.0], true) # real manifold does not allow complex values
    @test_throws DomainError is_vector(E, [1], [1.0, 1.0, 0.0], true)
    @test_throws DomainError is_vector(E, [0.0, 0.0, 0.0], [1.0], true)
    @test_throws DomainError is_vector(E, [0.0, 0.0, 0.0], [1.0, 0.0, 1.0im], true)
    @test_throws DomainError is_vector(Ec, [0.0, 0.0, 0.0], [:a, :b, :c], true)

    @test E^2 === Euclidean(3, 2)
    @test ^(E, 2) === Euclidean(3, 2)
    @test E^(2,) === Euclidean(3, 2)
    @test Ec^(4, 5) === Euclidean(3, 4, 5; field=ℂ)

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
                DiagonalizingOrthonormalBasis([1.0, 2.0, 3.0], ℂ),
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
                    test_reverse_diff=isa(T, Vector),
                    test_project_point=true,
                    test_project_tangent=true,
                    test_musical_isomorphisms=true,
                    test_default_vector_transport=true,
                    vector_transport_methods=[
                        ParallelTransport(),
                        SchildsLadderTransport(),
                        PoleLadderTransport(),
                    ],
                    test_mutating_rand=isa(T, Vector),
                    point_distributions=[
                        Manifolds.projected_distribution(
                            M,
                            Distributions.MvNormal(zero(pts[1]), 1.0),
                        ),
                    ],
                    tvector_distributions=[
                        Manifolds.normal_tvector_distribution(M, pts[1], 1.0),
                    ],
                    basis_types_vecs=basis_types,
                    basis_types_to_from=basis_types,
                    basis_has_specialized_diagonalizing_get=true,
                    test_vee_hat=isa(M, Euclidean),
                    test_inplace=true,
                    test_rand_point=M === E,
                    test_rand_tvector=M === E,
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
                test_reverse_diff=isa(T, Vector),
                test_project_tangent=true,
                test_musical_isomorphisms=true,
                test_default_vector_transport=true,
                test_vee_hat=false,
                parallel_transport=true,
            )
        end
    end

    number_types = [Float64, ComplexF64]
    TEST_FLOAT32 && push!(number_types, Float32)
    @testset "(Nonmutating) Real and Complex Numbers" begin
        RM = Euclidean()
        CM = Euclidean(; field=ℂ)
        for T in number_types
            @testset "Type $T" begin
                M = (T <: Complex) ? CM : RM
                pts = convert.(Ref(T), [1.0, 4.0, 2.0])
                @test embed(M, pts[1]) == pts[1]
                @test project(M, pts[1]) == pts[1]
                @test retract(M, pts[1], pts[2]) == exp(M, pts[1], pts[2])
                test_manifold(
                    M,
                    pts,
                    test_forward_diff=false,
                    test_reverse_diff=false,
                    test_vector_spaces=false,
                    test_project_tangent=true,
                    test_musical_isomorphisms=true,
                    test_default_vector_transport=true,
                    test_vee_hat=false,
                    is_mutating=false,
                )
            end
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

        @test ℂ^2 === Euclidean(2; field=ℂ)
        @test ℂ^(2, 3) === Euclidean(2, 3; field=ℂ)

        @test ℍ^2 === Euclidean(2; field=ℍ)
        @test ℍ^(2, 3) === Euclidean(2, 3; field=ℍ)
    end

    @testset "Embeddings into larger Euclidean Manifolds" begin
        M = Euclidean(3, 3)
        N = Euclidean(4, 4)
        O = EmbeddedManifold(M, N)
        # first test with same length of sizes
        p = ones(3, 3)
        q = zeros(4, 4)
        qT = zeros(4, 4)
        qT[1:3, 1:3] .= 1.0
        embed!(O, q, p)
        @test norm(qT - q) == 0
        qM = embed(O, p)
        @test norm(project(O, qM) - p) == 0
        @test norm(qT - qM) == 0
        # test with different sizes, check that it only fills first element
        q2 = zeros(4, 4, 3)
        q2T = zeros(4, 4, 3)
        q2T[1:3, 1:3, 1] .= 1.0
        embed!(O, q2, p)
        @test norm(q2T - q2) == 0
        O2 = EmbeddedManifold(M, Euclidean(4, 4, 3))
        q2M = embed(O2, p)
        @test norm(q2T - q2M) == 0
        # wrong size error checks
        @test_throws DomainError embed!(O, zeros(3, 3), zeros(3, 3, 5))
        @test_throws DomainError embed!(O, zeros(3, 3), zeros(4, 4))
        @test_throws DomainError project!(O, zeros(3, 3, 5), zeros(3, 3))
        @test_throws DomainError project!(O, zeros(4, 4), zeros(3, 3))
    end

    @testset "Embedding Real into Complex" begin
        M = Euclidean(3, 3)
        N = Euclidean(3, 4; field=ℂ)
        O = EmbeddedManifold(M, N)
        p = ones(3, 3)
        qT = zeros(ComplexF64, 3, 4)
        qT[1:3, 1:3] .= 1.0
        q = embed(O, p)
        @test is_point(N, q)
        @test q == qT
        q2 = zeros(ComplexF64, 3, 4)
        embed!(O, q2, p)
        @test q2 == qT
    end

    @testset "Euclidean metric tests" begin
        M = Euclidean(2)
        p = zeros(2)
        A = Manifolds.get_default_atlas(M)
        i = Manifolds.get_chart_index(M, A, p)
        B = Manifolds.induced_basis(M, A, i, TangentSpace)
        C1 = christoffel_symbols_first(M, p, B)
        @test size(C1) == (2, 2, 2)
        @test norm(C1) ≈ 0.0 atol = 1e-13
        C2 = christoffel_symbols_second(M, p, B)
        @test size(C2) == (2, 2, 2)
        @test norm(C2) ≈ 0.0 atol = 1e-13
        C2j = christoffel_symbols_second_jacobian(M, p, B)
        @test size(C2j) == (2, 2, 2, 2)
        @test norm(C2j) ≈ 0.0 atol = 1e-16
        @test einstein_tensor(M, p, B) == zeros(2, 2)
        @test ricci_curvature(M, p, B) ≈ 0 atol = 1e-16
        RC = ricci_tensor(M, p, B)
        @test size(RC) == (2, 2)
        @test norm(RC) ≈ 0.0 atol = 1e-16
        @test local_metric(M, p, B) == Diagonal(ones(2))
        @test inverse_local_metric(M, p, B) == Diagonal(ones(2))
        @test det_local_metric(M, p, B) == 1
        RT = riemann_tensor(M, p, B)
        @test size(RT) == (2, 2, 2, 2)
        @test norm(RT) ≈ 0.0 atol = 1e-16
    end
    @testset "Induced Basis and local metric for EuclideanMetric" begin
        struct DefaultManifold <: AbstractManifold{ℝ} end
        p = zeros(3)
        M = DefaultManifold()
        TpM = TangentSpace(M, p)
        B = induced_basis(M, Manifolds.get_default_atlas(M), p, TangentSpace)
        MM = MetricManifold(M, EuclideanMetric())
        @test local_metric(MM, p, B) == Diagonal(ones(3))
        @test inverse_local_metric(MM, p, B) == Diagonal(ones(3))
        @test det_local_metric(MM, p, B) == 1.0
        DB1 = dual_basis(MM, p, B)
        @test DB1 isa InducedBasis
        @test DB1.vs isa ManifoldsBase.CotangentSpaceType
        DB2 = induced_basis(M, Manifolds.get_default_atlas(M), p, CotangentSpace)
        @test DB2 isa InducedBasis
        @test DB2.vs isa ManifoldsBase.CotangentSpaceType
        DDB = dual_basis(MM, p, DB2)
        @test DDB isa InducedBasis
        @test DDB.vs isa ManifoldsBase.TangentSpaceType
    end
    @testset "RNG point with σ" begin
        @test is_point(E, rand(E; σ=10.0))
        @test is_point(E, rand(MersenneTwister(123), E; σ=10.0))
    end
end
