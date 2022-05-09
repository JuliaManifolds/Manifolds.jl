include("../utils.jl")

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
        @test !is_point(M, [1.0, 0.0, 0.0, 0.0])
        @test !is_vector(M, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0])
        @test_throws DomainError is_point(M, [2.0, 0.0, 0.0], true)
        @test !is_point(M, [2.0, 0.0, 0.0])
        @test !is_vector(M, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test_throws ManifoldDomainError is_vector(
            M,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            true,
        )
        @test !is_vector(M, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0])
        @test_throws DomainError is_vector(M, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], true)
        @test is_default_metric(M, MinkowskiMetric())
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
            # copyto
            pC = allocate(p)
            copyto!(M, pC, p)
            @test pC.value == p.value
            XC = allocate(X)
            @test copyto!(M, XC, p, X) == X # does copyto return the right value?
            @test XC == X # does copyto store the right value?
            @test XC.value == X.value # another check
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
        is_point(M, pH)
        pB = convert(PoincareBallPoint, p)
        @test pB.value == convert(PoincareBallPoint, pH).value
        @test is_point(M, pB)
        @test convert(AbstractVector, pB) == p # convert back yields again p
        @test convert(HyperboloidPoint, pB).value == pH.value
        @test_throws DomainError is_point(M, PoincareBallPoint([0.9, 0.0, 0.0]), true)
        @test_throws DomainError is_point(M, PoincareBallPoint([1.1, 0.0]), true)

        @test is_vector(M, pB, PoincareBallTVector([2.0, 2.0]))
        @test_throws DomainError is_vector(
            M,
            pB,
            PoincareBallTVector([2.0, 2.0, 3.0]),
            true,
        )
        pS = convert(PoincareHalfSpacePoint, p)
        pS2 = convert(PoincareHalfSpacePoint, pB)
        pS3 = convert(PoincareHalfSpacePoint, pH)

        @test_throws DomainError is_point(M, PoincareHalfSpacePoint([0.0, 0.0, 1.0]), true)
        @test_throws DomainError is_point(M, PoincareHalfSpacePoint([0.0, -1.0]), true)

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
            # Test broadcast
            @test 2 .* X1 == T(2 .* X1.value)
            @test copy(X1) == X1
            @test copy(X1) !== X1
            X1s = similar(X1)
            X1s .= 2 .* X1
            @test X1s == 2 * X1
            X1s .= X1
            @test X1s == X1

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
                test_tangent_vector_broadcasting=is_plain_array,
                test_vector_spaces=is_plain_array,
                test_inplace=true,
                test_rand_point=is_plain_array,
                test_rand_tvector=is_plain_array,
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
        q3 = similar(p)
        @test embed(M, p2) == p2.value
        embed!(M, q2, p2)
        embed!(M, q3, p2)
        @test q2.value == p2.value
        @test q3 == p2.value
        @test embed(M, p2, X2) == X2.value
        Y2 = HyperboloidTVector(similar(X))
        Y3 = similar(X)
        embed!(M, Y2, p2, X2)
        @test Y2.value == X2.value
        embed!(M, Y3, p2, X2)
        @test Y3 == X2.value
        # check embed for PoincareBall
        p4 = convert(PoincareBallPoint, p)
        X4 = convert(PoincareBallTVector, p, X)
        q4 = embed(M, p4)
        @test isapprox(q4, zeros(2))
        q4b = similar(q4)
        embed!(M, q4b, p4)
        @test q4b == q4
        Y4 = embed(M, p4, X4)
        @test Y4 == X4.value
        Y4b = similar(Y4)
        embed!(M, Y4b, p4, X4)
        @test Y4 == Y4b
        # check embed for PoincareHalfSpace
        p5 = convert(PoincareHalfSpacePoint, p)
        X5 = convert(PoincareHalfSpaceTVector, p, X)
        q5 = embed(M, p5)
        @test isapprox(q5, [0.0; 1.0])
        q5b = similar(q5)
        embed!(M, q5b, p5)
        @test q5b == q5
        Y5 = embed(M, p5, X5)
        @test Y5 == X5.value
        Y5b = similar(Y5)
        embed!(M, Y5b, p5, X5)
        @test Y5 == Y5b
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
            @test is_vector(M, p, v, true)
            for b in [DefaultOrthonormalBasis(), DiagonalizingOrthonormalBasis(V[1])]
                @test isapprox(M, p, v, get_vector(M, p, get_coordinates(M, p, v, b), b))
            end
        end
        for v in V, w in V
            @test inner(M, p, v, w) ≈ (v == w ? 1 : 0)
        end
        X = 0.5 * V[1] + 1.0 .* V[2]
        @test is_vector(M, p, X)
        c = get_coordinates(M, p, X, B)
        @test c ≈ [0.5, 1.0]
        c2 = similar(c)
        get_coordinates!(M, c2, p, X, DefaultOrthonormalBasis())
        B2 = DiagonalizingOrthonormalBasis(X)
        V2 = get_vectors(M, p, get_basis(M, p, B2))
        @test V2[1] ≈ X ./ norm(M, p, X)
        @test inner(M, p, V2[1], V2[2]) ≈ 0.0 atol = 5e-16
        B3 = DiagonalizingOrthonormalBasis(-V[2])
        V3 = get_vectors(M, p, get_basis(M, p, B3))
        @test isapprox(M, p, V3[1], -V[2])
    end
    @testset "Hyperboloid Minkowski Metric tests" begin
        p = Manifolds._hyperbolize(M, [0.0, 1.0])
        X = [1.0, 0.0, 0.0]
        pH = HyperboloidPoint(p)
        XH = HyperboloidTVector(X)
        @test minkowski_metric(XH, pH) == minkowski_metric(X, p)
        @test minkowski_metric(pH, XH) == minkowski_metric(p, X)
        @test minkowski_metric(XH, XH) == minkowski_metric(X, X)
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
        p2 = PoincareHalfSpacePoint([1.1, 1.1])
        X2 = PoincareHalfSpaceTVector([0.2, 0.2])
        Y2 = PoincareHalfSpaceTVector(allocate(X2.value))
        project!(M, Y2, p2, X2)
        @test Y2 == X2
        Z2 = project(M, p2, X2)
        @test isa(Z2, PoincareHalfSpaceTVector)
        @test Z2 == X2
    end
    @testset "Metric conversion on Hyperboloid" begin
        M = Hyperbolic(2)
        p = [1.0, 1.0, sqrt(3)]
        X = [1.0, 2.0, sqrt(3)]
        Y = change_representer(M, EuclideanMetric(), p, X)
        @test inner(M, p, X, Y) == inner(Euclidean(3), p, X, X)
        # change metric not possible from Euclidean, since the embedding is Lorenzian
        @test_throws ErrorException change_metric(M, EuclideanMetric(), p, X)
        # but if we come from the same metric, we have the identity
        @test change_metric(M, MinkowskiMetric(), p, X) == X
    end
    @testset "Metric conversion on Poincare Ball" begin
        M = Hyperbolic(2)
        p = convert(PoincareBallPoint, [1.0, 1.0, sqrt(3)])
        X = convert(PoincareBallTVector, [1.0, 1.0, sqrt(3)], [1.0, 2.0, sqrt(3)])
        Y = change_representer(M, EuclideanMetric(), p, X)
        @test inner(M, p, X, Y) == inner(Euclidean(3), p, X.value, X.value)
        α = 2 / (1 - norm(p.value)^2)
        @test Y.value == X.value ./ α^2
        Z = change_metric(M, EuclideanMetric(), p, X)
        @test Z.value == X.value ./ α
        A = change_metric(M, MinkowskiMetric(), p, X)
        @test A == X
    end
end
