include("../utils.jl")
include("group_utils.jl")

@testset "Special Orthogonal group" begin
    G = SpecialOrthogonal(3)
    @test repr(G) == "SpecialOrthogonal(3)"
    M = base_manifold(G)
    @test M === Rotations(3)
    x = Matrix(I, 3, 3)

    @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
    @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(true)
    @test (@inferred Manifolds.biinvariant_metric_dispatch(G)) === Val(true)
    @test is_default_metric(MetricManifold(G, EuclideanMetric())) === true
    @test is_default_metric(
        MetricManifold(G, InvariantMetric(EuclideanMetric(), LeftAction())),
    ) === true
    @test is_default_metric(
        MetricManifold(G, InvariantMetric(EuclideanMetric(), RightAction())),
    ) === true
    @test Manifolds.default_metric_dispatch(G, EuclideanMetric()) === Val{true}()
    @test Manifolds.default_metric_dispatch(MetricManifold(G, EuclideanMetric())) ===
          Val{true}()

    types = [Matrix{Float64}]
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    pts = [exp(M, x, hat(M, x, ωi)) for ωi in ω]
    vpts = [hat(M, x, [-1.0, 2.0, 0.5]), hat(M, x, [1.0, 0.0, 0.5])]

    ge = allocate(pts[1])
    copyto!(ge, make_identity(G, pts[1]))
    @test isapprox(ge, I; atol=1e-10)

    gI = Identity(G, ge)
    gT = allocate_result(G, exp, gI, log(G, pts[1], pts[2]))
    @test size(gT) == size(ge)
    @test eltype(gT) == eltype(ge)
    gT = allocate_result(G, log, gI, pts[1])
    @test size(gT) == size(ge)
    @test eltype(gT) == eltype(ge)

    for T in types
        gpts = convert.(T, pts)
        vgpts = convert.(T, vpts)
        @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] * gpts[2]
        @test translate_diff(G, gpts[2], gpts[1], vgpts[1], LeftAction()) ≈ vgpts[1]
        @test translate_diff(G, gpts[2], gpts[1], vgpts[1], RightAction()) ≈
              transpose(gpts[2]) * vgpts[1] * gpts[2]
        test_group(G, gpts, vgpts, vgpts; test_diff=true, test_invariance=true)
    end

    @testset "Decorator forwards to group" begin
        DM = NotImplementedGroupDecorator(G)
        @test (@inferred Manifolds.decorator_group_dispatch(DM)) === Val(true)
        @test Manifolds.is_group_decorator(DM)
        @test base_group(DM) === G
        @test_throws DomainError is_point(
            DM,
            make_identity(TranslationGroup(3), [1, 2, 3]),
            true,
        )
        test_group(DM, pts, vpts, vpts; test_diff=true)
    end

    @testset "Group forwards to decorated" begin
        retraction_methods = [
            Manifolds.PolarRetraction(),
            Manifolds.QRRetraction(),
            Manifolds.GroupExponentialRetraction(LeftAction()),
            Manifolds.GroupExponentialRetraction(RightAction()),
        ]

        inverse_retraction_methods = [
            Manifolds.PolarInverseRetraction(),
            Manifolds.QRInverseRetraction(),
            Manifolds.GroupLogarithmicInverseRetraction(LeftAction()),
            Manifolds.GroupLogarithmicInverseRetraction(RightAction()),
        ]

        test_manifold(
            G,
            pts;
            test_reverse_diff=false,
            test_injectivity_radius=false,
            test_project_tangent=true,
            test_musical_isomorphisms=false,
            retraction_methods=retraction_methods,
            inverse_retraction_methods=inverse_retraction_methods,
            exp_log_atol_multiplier=20,
            retraction_atol_multiplier=12,
            is_tangent_atol_multiplier=1.2,
            test_atlases=(Manifolds.RetractionAtlas(),),
        )

        @test injectivity_radius(G) == injectivity_radius(M)
        @test injectivity_radius(G, pts[1]) == injectivity_radius(M, pts[1])
        @test injectivity_radius(G, pts[1], PolarRetraction()) ==
              injectivity_radius(M, pts[1], PolarRetraction())

        y = allocate(pts[1])
        exp!(G, y, pts[1], vpts[1])
        @test isapprox(M, y, exp(M, pts[1], vpts[1]))

        y = allocate(pts[1])
        retract!(G, y, pts[1], vpts[1])
        @test isapprox(M, y, retract(M, pts[1], vpts[1]))

        w = allocate(vpts[1])
        inverse_retract!(G, w, pts[1], pts[2])
        @test isapprox(M, pts[1], w, inverse_retract(M, pts[1], pts[2]))

        w = allocate(vpts[1])
        inverse_retract!(G, w, pts[1], pts[2], QRInverseRetraction())
        @test isapprox(
            M,
            pts[1],
            w,
            inverse_retract(M, pts[1], pts[2], QRInverseRetraction()),
        )

        z = collect(reshape(1.0:9.0, 3, 3))
        @test isapprox(M, project(G, z), project(M, z))
        z2 = allocate(pts[1])
        project!(G, z2, z)
        @test isapprox(M, z2, project(M, z))
    end

    @testset "vee/hat" begin
        X = vpts[1]
        pe = identity(G, pts[1])

        Xⁱ = vee(G, make_identity(G, pts[1]), X)
        @test Xⁱ ≈ vee(G, pe, X)

        X2 = hat(G, make_identity(G, pts[1]), Xⁱ)
        @test isapprox(M, pe, X2, hat(G, pe, Xⁱ); atol=1e-6)
    end
    @testset "Identity and get_vector/get_coordinates" begin
        e = Identity(G, Matrix{Float64}(I, 3, 3))
        gT = allocate_result(G, get_coordinates, e, pts[1])
        @test size(gT) == (manifold_dimension(M),)
        @test eltype(gT) == eltype(e.p)
        @test_throws ErrorException allocate_result(M, get_vector, e, pts[1])
        gT = allocate_result(G, get_vector, e, pts[1])
        @test size(gT) == size(e.p)
        @test eltype(gT) == eltype(e.p)
        eT = similar(e.p)
        copyto!(eT, e)
        @test eT == e.p

        eF = Identity(SpecialEuclidean(3), 1)
        c = [1.0, 0.0, 0.0]
        Y = zeros(representation_size(G))
        get_vector!(G, Y, e, c, Manifolds.VeeOrthogonalBasis())
        @test Y ≈ get_vector(decorated_manifold(G), e.p, c, Manifolds.VeeOrthogonalBasis())
        @test_throws ErrorException get_vector!(G, Y, eF, c, Manifolds.VeeOrthogonalBasis())
        get_vector!(M, Y, e, c, Manifolds.VeeOrthogonalBasis())
        @test Y ≈ get_vector(decorated_manifold(G), e.p, c, Manifolds.VeeOrthogonalBasis())
        @test_throws ErrorException get_vector!(M, Y, eF, c, Manifolds.VeeOrthogonalBasis())

        @test get_coordinates(
            decorated_manifold(G),
            e,
            Y,
            Manifolds.VeeOrthogonalBasis(),
        ) == c
        @test_throws ErrorException get_coordinates(
            M,
            eF,
            c,
            Manifolds.VeeOrthogonalBasis(),
        )
        c2 = similar(c)
        get_coordinates!(G, c2, e, Y, Manifolds.VeeOrthogonalBasis())
        @test c == c2
        @test_throws ErrorException get_coordinates!(
            G,
            c2,
            eF,
            Y,
            Manifolds.VeeOrthogonalBasis(),
        )
        get_coordinates!(M, c2, e, Y, Manifolds.VeeOrthogonalBasis())
        @test c == c2
        @test_throws ErrorException get_coordinates!(
            M,
            c2,
            eF,
            Y,
            Manifolds.VeeOrthogonalBasis(),
        )
    end
end
