include("../utils.jl")
include("group_utils.jl")

@testset "Special Orthogonal group" begin
    for n in [2, 3]
        G = SpecialOrthogonal(n)
        @test repr(G) == "SpecialOrthogonal($n)"
        M = base_manifold(G)
        @test M === Rotations(n)
        p = Matrix(I, n, n)
        @test is_default_metric(MetricManifold(G, EuclideanMetric()))

        types = [Matrix{Float64}]

        if n == 2
            ω = [[1.0], [-1.0], [3.0]]
            pts = [exp(M, p, hat(M, p, ωi)) for ωi in ω]
            Xpts = [hat(M, p, [-1.5]), hat(M, p, [0.5])]
        else
            ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            pts = [exp(M, p, hat(M, p, ωi)) for ωi in ω]
            Xpts = [hat(M, p, [-1.0, 2.0, 0.5]), hat(M, p, [1.0, 0.0, 0.5])]
        end

        ge = allocate(pts[1])
        identity_element!(G, ge)
        @test isapprox(ge, I; atol=1e-10)

        gI = Identity(G)
        gT = allocate_result(G, exp, gI, log(G, pts[1], pts[2]))
        @test size(gT) == size(ge)
        @test eltype(gT) == eltype(ge)
        gT = allocate_result(G, log, gI, pts[1])
        @test size(gT) == size(ge)
        @test eltype(gT) == eltype(ge)

        for T in types
            gpts = convert.(T, pts)
            Xgpts = convert.(T, Xpts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] * gpts[2]
            @test translate_diff(G, gpts[2], gpts[1], Xgpts[1], LeftAction()) ≈ Xgpts[1]
            @test translate_diff(G, gpts[2], gpts[1], Xgpts[1], RightAction()) ≈
                  transpose(gpts[2]) * Xgpts[1] * gpts[2]
            test_group(
                G,
                gpts,
                Xgpts,
                Xgpts;
                test_diff=true,
                test_invariance=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
            )

            @testset "log_lie edge cases" begin
                X = Manifolds.hat(G, ge, vcat(Float64(π), zeros(manifold_dimension(G) - 1)))
                p = exp_lie(G, X)
                @test isapprox(p, exp_lie(G, log_lie(G, p)))
            end
        end

        @testset "Decorator forwards to group" begin
            DM = NotImplementedGroupDecorator(G)
            test_group(DM, pts, Xpts, Xpts; test_diff=true)
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
            exp!(G, y, pts[1], Xpts[1])
            @test isapprox(M, y, exp(M, pts[1], Xpts[1]))

            y = allocate(pts[1])
            retract!(G, y, pts[1], Xpts[1])
            @test isapprox(M, y, retract(M, pts[1], Xpts[1]))

            w = allocate(Xpts[1])
            inverse_retract!(G, w, pts[1], pts[2])
            @test isapprox(M, pts[1], w, inverse_retract(M, pts[1], pts[2]))

            w = allocate(Xpts[1])
            inverse_retract!(G, w, pts[1], pts[2], QRInverseRetraction())
            @test isapprox(
                M,
                pts[1],
                w,
                inverse_retract(M, pts[1], pts[2], QRInverseRetraction()),
            )

            if n == 2
                z = collect(reshape(1.0:4.0, 2, 2))
            else
                z = collect(reshape(1.0:9.0, 3, 3))
            end

            @test isapprox(M, project(G, z), project(M, z))
            z2 = allocate(pts[1])
            project!(G, z2, z)
            @test isapprox(M, z2, project(M, z))
        end

        @testset "vee/hat" begin
            X = Xpts[1]
            pe = Identity(G)

            Xⁱ = vee(G, identity_element(G), X)
            @test Xⁱ ≈ vee(G, pe, X)

            X2 = hat(G, pts[1], Xⁱ)
            @test isapprox(M, pe, X2, hat(G, pe, Xⁱ); atol=1e-6)
        end
        @testset "Identity and get_vector/get_coordinates" begin
            e = Identity(G)

            eF = Identity(SpecialEuclidean(n))
            if n == 2
                c = [1.0]
            else
                c = [1.0, 0.0, 0.0]
            end
            Y = zeros(representation_size(G))
            get_vector_lie!(G, Y, c, Manifolds.VeeOrthogonalBasis())
            @test Y ≈ get_vector_lie(G, c, Manifolds.VeeOrthogonalBasis())
            get_vector!(M, Y, identity_element(G), c, Manifolds.VeeOrthogonalBasis())
            @test Y ≈ get_vector(M, identity_element(G), c, Manifolds.VeeOrthogonalBasis())

            @test get_coordinates(
                M,
                identity_element(G),
                Y,
                Manifolds.VeeOrthogonalBasis(),
            ) == c
            c2 = similar(c)
            get_coordinates_lie!(G, c2, Y, Manifolds.VeeOrthogonalBasis())
            @test c == c2
            get_coordinates!(M, c2, identity_element(G), Y, Manifolds.VeeOrthogonalBasis())
            @test c == c2

            q = zeros(n, n)
            mul!(q, e, e)
            @test isone(q)
            e2 = Identity(G)
            @test mul!(e2, e, e) === e2
        end
    end
end
