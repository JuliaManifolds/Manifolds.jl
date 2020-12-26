include("utils.jl")

@testset "Hyperbolic Space" begin
    M = Hyperbolic(2)
    @testset "Hyperbolic Basics" begin
        @test repr(M) == "Hyperbolic(2)"
        @test base_manifold(M) == M
        @test manifold_dimension(M) == 2
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
        @test Manifolds.default_metric_dispatch(M, MinkowskiMetric()) === Val{true}()
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

        for (P, T) in zip(
            [HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint],
            [HyperboloidTVector, PoincareBallTVector, PoincareHalfSpaceTVector],
        )
            p = convert(P, [1.0, 0.0, sqrt(2.0)])
            X = convert(T, [1.0, 0.0, sqrt(2.0)], [0.0, 1.0, 0.0])
            @test number_eltype(p) == eltype(p.value)
            @test X * 2.0 == T(X.value * 2.0)
            @test 2 \ X == T(2 \ X.value)
            @test +X == T(+X.value)
            @test Manifolds.allocate_result_type(M, log, (p, p)) == T
            @test Manifolds.allocate_result_type(M, inverse_retract, (p, p)) == T
            convert(T, p, X) == X
        end
    end
    @testset "Hyperbolic Representation Conversion I" begin
        p = [0.0, 0.0, 1.0]
        pH = HyperboloidPoint(p)
        @test minkowski_metric(pH, pH) == minkowski_metric(p, p)
        @test convert(HyperboloidPoint, p).value == pH.value
        @test convert(AbstractVector, pH) == p
        X = [1.0, 0.0, 0.0]
        XH = HyperboloidTVector(X)

        @test convert(HyperboloidTVector, X).value == XH.value
        @test convert(AbstractVector, XH) == X
        @test convert(HyperboloidPoint, p).value == pH.value
        is_manifold_point(M, pH)
        pB = convert(PoincareBallPoint, p)
        @test pB.value == convert(PoincareBallPoint, pH).value
        @test is_manifold_point(M, pB)
        @test convert(AbstractVector, pB) == p # convert back yields again p
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
        @test convert(AbstractVector, pS) == convert(HyperboloidPoint, pS).value
        @test convert(PoincareBallPoint, pS2).value == pB.value
    end
    @testset "Hyperbolic Representation Conversion II" begin
        M = Hyperbolic(2)
        pts = [[0.0, 0.0, 1.0], [1.0, 0.0, sqrt(2.0)]]
        X = log(M, pts[2], pts[1])
        # For HyperboloidTVector we can do a plain wrap/unwrap
        X1 = convert(HyperboloidTVector, X)
        @test convert(AbstractVector, X1) == X
        # Convert to types and back to Array
        for (P, T) in zip(
            [HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint],
            [HyperboloidTVector, PoincareBallTVector, PoincareHalfSpaceTVector],
        )
            # convert to P,T
            p1 = convert(P, pts[2])
            X1 = convert(T, pts[2], X)
            (p2, X2) = convert(Tuple{P,T}, (pts[2], X))
            @test isapprox(M, p1, p2)
            @test isapprox(M, p1, X1, X2)
            for (P2, T2) in zip(
                [HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint],
                [HyperboloidTVector, PoincareBallTVector, PoincareHalfSpaceTVector],
            )
                @test isapprox(M, convert(P2, p1), convert(P2, pts[2]))
                @test convert(T, p1, X1) == convert(T, pts[2], X)
                (p3, X3) = convert(Tuple{P2,T2}, (pts[2], X))
                (p3a, X3a) = convert(Tuple{P2,T2}, (p1, X1))
                @test isapprox(M, p3, p3a)
                @test isapprox(M, p3, X3, X3a)
                @test isapprox(M, convert(P2, p2), p3)
                @test isapprox(M, pts[2], convert(AbstractVector, p3))
                @test isapprox(M, p3, convert(T2, p2, X2), X3)
                # coupled conversion
                (pT, XT) = convert(Tuple{AbstractVector,AbstractVector}, (p2, X2))
                @test isapprox(M, pts[2], pT)
                @test isapprox(M, pts[2], X, XT)
            end
        end
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
            is_plain_array =
                T ∉ [HyperboloidPoint, PoincareBallPoint, PoincareHalfSpacePoint]
            pts = [
                convert(T, [0.0, 0.0, 1.0]),
                convert(T, [1.0, 0.0, sqrt(2.0)]),
                convert(T, [0.0, 1.0, sqrt(2.0)]),
            ]
            test_manifold(
                M,
                pts,
                test_project_tangent=is_plain_array || T == HyperboloidPoint,
                test_musical_isomorphisms=is_plain_array,
                test_default_vector_transport=true,
                vector_transport_methods=[
                    ParallelTransport(),
                    SchildsLadderTransport(),
                    PoleLadderTransport(),
                ],
                is_tangent_atol_multiplier=10.0,
                exp_log_atol_multiplier=10.0,
                retraction_methods=(ExponentialRetraction(),),
                test_vee_hat=false,
                test_forward_diff=is_plain_array,
                test_reverse_diff=is_plain_array,
                test_tangent_vector_broadcasting=is_plain_array,
                test_vector_spaces=is_plain_array,
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
        p2 = HyperboloidPoint(p)
        X2 = HyperboloidTVector(X)
        q2 = HyperboloidPoint(similar(p))
        @test embed(M, p2).value == p2.value
        embed!(M, q2, p2)
        @test q2.value == p2.value
        @test embed(M, p2, X2).value == X2.value
        Y2 = HyperboloidTVector(similar(X))
        embed!(M, Y2, p2, X2)
        @test Y2.value == X2.value
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
        @test isapprox(M, mean(M, pts), pts[1]; atol=10^-4)
        @test isapprox(M, mean(M, pts, ws), pts[1]; atol=10^-4)
        @test_throws DimensionMismatch mean(M, pts, UnitWeights{Float64}(length(pts) + 1))
    end
    @testset "Hyperbolic ONB test" begin
        M = Hyperbolic(2)
        p = Manifolds._hyperbolize(M, [1.0, 0.0])
        B = get_basis(M, p, DefaultOrthonormalBasis())
        V = get_vectors(M, p, B)
        for v in V
            @test is_tangent_vector(M, p, v, true)
            for b in [DefaultOrthonormalBasis(), DiagonalizingOrthonormalBasis(V[1])]
                @test isapprox(M, p, v, get_vector(M, p, get_coordinates(M, p, v, b), b))
            end
        end
        for v in V, w in V
            @test inner(M, p, v, w) ≈ (v == w ? 1 : 0)
        end
        X = 0.5 * V[1] + 1.0 .* V[2]
        @test is_tangent_vector(M, p, X)
        c = get_coordinates(M, p, X, B)
        @test c ≈ [0.5, 1.0]
        B2 = DiagonalizingOrthonormalBasis(X)
        V2 = get_vectors(M, p, get_basis(M, p, B2))
        @test V2[1] ≈ X ./ norm(M, p, X)
        @test inner(M, p, V2[1], V2[2]) ≈ 0.0 atol = 5e-16
        B3 = DiagonalizingOrthonormalBasis(-V[2])
        V3 = get_vectors(M, p, get_basis(M, p, B3))
        @test isapprox(M, p, V3[1], -V[2])
    end
    @testset "Hyperboloid Minkowski Metric tests" begin
        p = Manifolds._hyperbolize(M,[0.0,1.0])
        X = [1.0, 0.0, 0.0]
        pH = HyperboloidPoint(p)
        XH = HyperboloidTVector(X)
        @test minkowski_metric(XH,pH) == minkowski_metric(X,p)
        @test minkowski_metric(pH,XH) == minkowski_metric(p,X)
        @test minkowski_metric(XH,XH) == minkowski_metric(X,X)
    end
    @testset "PoincareBall project tangent test" begin
        p = PoincareBallPoint([0.5, 0.0])
        X = PoincareBallTVector([0.2, 0.2])
        Y = PoincareBallTVector(allocate(X.value))
        project!(M, Y, p, X)
        @test Y == X
        Z = project(M, p, X)
        @test isa(Z, PoincareBallTVector)
        @test Z == X
        p2 = PoincareHalfSpacePoint([1.1,1.1])
        X2 = PoincareHalfSpaceTVector([0.2, 0.2])
        Y2 = PoincareHalfSpaceTVector(allocate(X2.value))
        project!(M, Y2, p2, X2)
        @test Y2 == X2
        Z2 = project(M, p2, X2)
        @test isa(Z2, PoincareHalfSpaceTVector)
        @test Z2 == X2
    end
end
