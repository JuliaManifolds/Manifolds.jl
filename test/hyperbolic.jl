include("utils.jl")

@testset "Hyperbolic Space" begin
    M = Hyperbolic(2)
    @testset "Hyperbolic Basics" begin
        @test repr(M) == "Hyperbolic(2)"
        @test base_manifold(M) == M
        @test typeof(get_embedding(M)) ==
              MetricManifold{ℝ,Euclidean{Tuple{3},ℝ},MinkowskiMetric}
        @test representation_size(M) == (3,)
        @test isinf(injectivity_radius(M))
        @test isinf(injectivity_radius(M, ExponentialRetraction()))
        @test isinf(injectivity_radius(M, [0.0, 0.0, 1.0]))
        @test isinf(injectivity_radius(M, [0.0, 0.0, 1.0], ExponentialRetraction()))
        @test !is_manifold_point(M, [1.0, 0.0, 0.0, 0.0])
        @test !is_tangent_vector(M, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0])
        @test_throws DomainError is_manifold_point(M, [2.0, 0.0, 0.0], true)
        @test !is_manifold_point(M, [2.0, 0.0, 0.0])
        @test !is_tangent_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws DomainError is_tangent_vector(
            M,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            true,
        )
        @test !is_tangent_vector(M, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        @test_throws DomainError is_tangent_vector(
            M,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            true,
        )
        @test is_default_metric(M, MinkowskiMetric())
        @test Manifolds.default_metric_dispatch(M, MinkowskiMetric()) === Val{true}()
        @test manifold_dimension(M) == 2
    end
    @testset "Hyperbolic Representation Conversion" begin
        p = [0.0, 0.0, 1.0]
        pH = HyperboloidPoint(p)
        @test convert(HyperboloidPoint, p).value == pH.value
        @test convert(Vector, pH) == p
        X = [1.0, 0.0, 0.0]
        XH = HyperboloidTVector(X)
        @test convert(HyperboloidTVector, X).value == XH.value
        @test convert(Vector, XH) == X
        @test convert(HyperboloidPoint, p).value == pH.value
        is_manifold_point(M, pH)
        pB = convert(PoincareBallPoint, p)
        @test pB.value == convert(PoincareBallPoint, pH).value
        @test is_manifold_point(M, pB)
        @test convert(Vector, pB) == p # convert back yields again p
        @test convert(HyperboloidPoint, pB).value == pH.value
        @test_throws DomainError is_manifold_point(
            M,
            PoincareBallPoint([0.9, 0.0, 0.0]),
            true,
        )
        @test_throws DomainError is_manifold_point(M, PoincareBallPoint([1.0, 0.0]), true)

        @test is_tangent_vector(M, pB, PoincareBallTVector([2.0, 2.0]))

        pS = convert(PoincareHalfSpacePoint, p)
        pS2 = convert(PoincareHalfSpacePoint, pB)
        pS3 = convert(PoincareHalfSpacePoint, pH)

        @test_throws DomainError is_manifold_point(
            M,
            PoincareHalfSpacePoint([0.0, 0.0, 1.0]),
            true,
        )
        @test_throws DomainError is_manifold_point(
            M,
            PoincareHalfSpacePoint([0.0, -1.0]),
            true,
        )

        @test pS.value == pS2.value
        @test pS.value == pS3.value
        @test convert(Vector, pS) == convert(HyperboloidPoint, pS).value
        @test convert(PoincareBallPoint, pS2).value == pB.value
    end


    types = [
        Vector{Float64},
        SizedVector{3,Float64},
        HyperboloidPoint,
        PoincareBallPoint,
        PoincareHalfSpacePoint,
    ]
    TEST_FLOAT32 && push!(types, Vector{Float32})
    for T in types
        @testset "Type $T" begin
            pts = [
                convert(T, [0.0, 0.0, 1.0]),
                convert(T, [1.0, 0.0, sqrt(2.0)]),
                convert(T, [0.0, 1.0, sqrt(2.0)]),
            ]
            test_manifold(
                M,
                pts,
                test_project_tangent = true,
                test_musical_isomorphisms = true,
                test_default_vector_transport = true,
                vector_transport_methods = [
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                is_tangent_atol_multiplier = 10.0,
                exp_log_atol_multiplier = 10.0,
                retraction_methods = (ExponentialRetraction(),),
                test_vee_hat = false,
                test_forward_diff = !(
                    T ∉ [HyperboloidPoint, PoincareBallPoint, PoincareBallPoint]
                ),
                test_reverse_diff = !(
                    T ∉ [HyperboloidPoint, PoincareBallPoint, PoincareBallPoint]
                ),
            )
        end
    end
    @testset "Embedding test" begin
        p = [0.0, 0.0, 1.0]
        X = [1.0, 1.0, 0.0]
        @test embed(M, p) == p
        q = similar(p)
        embed!(M, q, p)
        @test q == p
        @test embed(M, p, X) == X
        Y = similar(X)
        embed!(M, Y, p, X)
        @test Y == X
    end
    @testset "Hyperbolic mean test" begin
        pts = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, sqrt(2.0)],
            [-1.0, 0.0, sqrt(2.0)],
            [0.0, 1.0, sqrt(2.0)],
            [0.0, -1.0, sqrt(2.0)],
        ]
        ws = UnitWeights{Float64}(length(pts))
        @test isapprox(M, mean(M, pts), pts[1]; atol = 10^-4)
        @test isapprox(M, mean(M, pts, ws), pts[1]; atol = 10^-4)
        @test_throws DimensionMismatch mean(M, pts, UnitWeights{Float64}(length(pts) + 1))
    end
end
