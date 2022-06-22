include("../utils.jl")
include("group_utils.jl")

using ManifoldsBase: VeeOrthogonalBasis

Random.seed!(10)

@testset "Special Euclidean group" begin
    @testset "SpecialEuclidean($n)" for n in (2, 3, 4)
        G = SpecialEuclidean(n)
        @test isa(G, SpecialEuclidean{n})
        @test repr(G) == "SpecialEuclidean($n)"
        M = base_manifold(G)
        @test M === TranslationGroup(n) × SpecialOrthogonal(n)
        @test submanifold(G, 1) === TranslationGroup(n)
        @test submanifold(G, 2) === SpecialOrthogonal(n)
        Rn = Rotations(n)
        p = Matrix(I, n, n)

        if n == 2
            t = Vector{Float64}.([1:2, 2:3, 3:4])
            ω = [[1.0], [2.0], [1.0]]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = [([-1.0, 2.0], hat(Rn, p, [1.0])), ([1.0, -2.0], hat(Rn, p, [0.5]))]
        elseif n == 3
            t = Vector{Float64}.([1:3, 2:4, 4:6])
            ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = [
                ([-1.0, 2.0, 1.0], hat(Rn, p, [1.0, 0.5, -0.5])),
                ([-2.0, 1.0, 0.5], hat(Rn, p, [-1.0, -0.5, 1.1])),
            ]
        else # n == 4
            t = Vector{Float64}.([1:4, 2:5, 3:6])
            ω = [
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                [1.0, 3.0, 2.0, 1.0, 2.0, 3.0],
            ]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = [
                ([-1.0, 2.0, 1.0, 3.0], hat(Rn, p, [1.0, 0.5, -0.5, 0.0, 2.0, 1.0])),
                ([-2.0, 1.5, -1.0, 2.0], hat(Rn, p, [1.0, -0.5, 0.5, 1.0, 0.0, 1.0])),
            ]
        end

        basis_types = (DefaultOrthonormalBasis(),)

        @testset "product repr" begin
            pts = [ProductRepr(tp...) for tp in tuple_pts]
            X_pts = [ProductRepr(tX...) for tX in tuple_X]

            g1, g2 = pts[1:2]
            t1, R1 = g1.parts
            t2, R2 = g2.parts
            g1g2 = ProductRepr(R1 * t2 + t1, R1 * R2)
            @test isapprox(G, compose(G, g1, g2), g1g2)
            g1g2mat = affine_matrix(G, g1g2)
            @test g1g2mat ≈ affine_matrix(G, g1) * affine_matrix(G, g2)
            @test affine_matrix(G, g1g2mat) === g1g2mat
            @test affine_matrix(G, Identity(G)) isa SDiagonal{n,Float64}
            @test affine_matrix(G, Identity(G)) == SDiagonal{n,Float64}(I)

            w = translate_diff(G, pts[1], Identity(G), X_pts[1])
            w2 = allocate(w)
            w2.parts[1] .= w.parts[1]
            w2.parts[2] .= pts[1].parts[2] * w.parts[2]
            w2mat = screw_matrix(G, w2)
            @test w2mat ≈ affine_matrix(G, pts[1]) * screw_matrix(G, X_pts[1])
            @test screw_matrix(G, w2mat) === w2mat

            test_group(
                G,
                pts,
                X_pts,
                X_pts;
                test_diff=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
                diff_convs=[(), (LeftAction(),), (RightAction(),)],
            )
            test_manifold(
                G,
                pts;
                basis_types_vecs=basis_types,
                basis_types_to_from=basis_types,
                is_mutating=true,
                #test_inplace=true,
                test_vee_hat=true,
                exp_log_atol_multiplier=50,
            )

            for CS in [CartanSchoutenMinus(), CartanSchoutenPlus(), CartanSchoutenZero()]
                @testset "$CS" begin
                    G_TR = ConnectionManifold(G, CS)

                    test_group(
                        G_TR,
                        pts,
                        X_pts,
                        X_pts;
                        test_diff=true,
                        test_lie_bracket=true,
                        test_adjoint_action=true,
                        diff_convs=[(), (LeftAction(),), (RightAction(),)],
                    )

                    test_manifold(
                        G_TR,
                        pts;
                        is_mutating=true,
                        exp_log_atol_multiplier=50,
                        test_inner=false,
                        test_norm=false,
                    )
                end
            end
            for MM in [LeftInvariantMetric()]
                @testset "$MM" begin
                    G_TR = MetricManifold(G, MM)

                    test_group(
                        G_TR,
                        pts,
                        X_pts,
                        X_pts;
                        test_diff=true,
                        test_lie_bracket=true,
                        test_adjoint_action=true,
                        diff_convs=[(), (LeftAction(),), (RightAction(),)],
                    )

                    test_manifold(
                        G_TR,
                        pts;
                        basis_types_vecs=basis_types,
                        basis_types_to_from=basis_types,
                        is_mutating=true,
                        exp_log_atol_multiplier=50,
                    )
                end
            end
        end

        @testset "affine matrix" begin
            pts = [affine_matrix(G, ProductRepr(tp...)) for tp in tuple_pts]
            X_pts = [screw_matrix(G, ProductRepr(tX...)) for tX in tuple_X]
            test_group(
                G,
                pts,
                X_pts,
                X_pts;
                test_diff=true,
                test_lie_bracket=true,
                diff_convs=[(), (LeftAction(),), (RightAction(),)],
                atol=1e-9,
            )
            test_manifold(
                G,
                pts;
                is_mutating=true,
                #test_inplace=true,
                test_vee_hat=true,
                exp_log_atol_multiplier=50,
            )
            # specific affine tests
            p = copy(G, pts[1])
            X = copy(G, p, X_pts[1])
            X[n + 1, n + 1] = 0.1
            @test_throws DomainError is_vector(G, p, X, true)
            X2 = zeros(n + 2, n + 2)
            # nearly correct just too large (and the error from before)
            X2[1:n, 1:n] .= X[1:n, 1:n]
            X2[1:n, end] .= X[1:n, end]
            X2[end, end] = X[end, end]
            @test_throws DomainError is_vector(G, p, X2, true)
            p[n + 1, n + 1] = 0.1
            @test_throws DomainError is_point(G, p, true)
            p2 = zeros(n + 2, n + 2)
            # nearly correct just too large (and the error from before)
            p2[1:n, 1:n] .= p[1:n, 1:n]
            p2[1:n, end] .= p[1:n, end]
            p2[end, end] = p[end, end]
            @test_throws DomainError is_point(G, p2, true)
            # exp/log_lie for ProductGroup on arrays
            X = copy(G, p, X_pts[1])
            p3 = exp_lie(G, X)
            X3 = log_lie(G, p3)
            isapprox(G, Identity(G), X, X3)
        end

        @testset "hat/vee" begin
            p = ProductRepr(tuple_pts[1]...)
            X = ProductRepr(tuple_X[1]...)
            Xexp = [X.parts[1]; vee(Rn, p.parts[2], X.parts[2])]
            Xc = vee(G, p, X)
            @test Xc ≈ Xexp
            @test isapprox(G, p, hat(G, p, Xc), X)

            Xc = vee(G, affine_matrix(G, p), screw_matrix(G, X))
            @test Xc ≈ Xexp
            @test hat(G, affine_matrix(G, p), Xc) ≈ screw_matrix(G, X)

            e = Identity(G)
            Xe = log_lie(G, p)
            Xc = vee(G, e, Xe)
            @test_throws MethodError vee(M, e, Xe)
            w = similar(Xc)
            vee!(G, w, e, Xe)
            @test isapprox(Xc, w)
            @test_throws MethodError vee!(M, w, e, Xe)

            w = similar(Xc)
            vee!(G, w, identity_element(G), Xe)
            @test isapprox(Xc, w)

            Ye = hat(G, e, Xc)
            @test_throws MethodError hat(M, e, Xc)
            isapprox(G, e, Xe, Ye)
            Ye2 = copy(G, p, X)
            hat!(G, Ye2, e, Xc)
            @test_throws MethodError hat!(M, Ye, e, Xc)
            @test isapprox(G, e, Ye, Ye2)

            Ye2 = copy(G, p, X)
            hat!(G, Ye2, identity_element(G), Xc)
            @test isapprox(G, e, Ye, Ye2)
        end
    end

    G = SpecialEuclidean(11)
    @test affine_matrix(G, Identity(G)) isa Diagonal{Float64,Vector{Float64}}
    @test affine_matrix(G, Identity(G)) == Diagonal(ones(11))

    @testset "Explicit embedding in GL(n+1)" begin
        G = SpecialEuclidean(3)
        t = Vector{Float64}.([1:3, 2:4, 4:6])
        ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
        p = Matrix(I, 3, 3)
        Rn = Rotations(3)
        pts = [ProductRepr(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
        X = ProductRepr([-1.0, 2.0, 1.0], hat(Rn, p, [1.0, 0.5, -0.5]))
        q = ProductRepr([0.0, 0.0, 0.0], p)

        GL = GeneralLinear(4)
        SEGL = EmbeddedManifold(G, GL)
        @test Manifolds.SpecialEuclideanInGeneralLinear(3) === SEGL
        pts_gl = [embed(SEGL, pp) for pp in pts]
        q_gl = embed(SEGL, q)
        X_gl = embed(SEGL, pts_gl[1], X)

        q_gl2 = allocate(q_gl)
        embed!(SEGL, q_gl2, q)
        @test isapprox(SEGL, q_gl2, q_gl)

        q2 = allocate(q)
        project!(SEGL, q2, q_gl)
        @test isapprox(G, q, q2)

        @test isapprox(G, pts[1], project(SEGL, pts_gl[1]))
        @test isapprox(G, pts[1], X, project(SEGL, pts_gl[1], X_gl))

        X_gl2 = allocate(X_gl)
        embed!(SEGL, X_gl2, pts_gl[1], X)
        @test isapprox(SEGL, pts_gl[1], X_gl2, X_gl)

        X2 = allocate(X)
        project!(SEGL, X2, pts_gl[1], X_gl)
        @test isapprox(G, pts[1], X, X2)

        for conv in [LeftAction(), RightAction()]
            tpgl = translate(GL, pts_gl[2], pts_gl[1], conv)
            tXgl = translate_diff(GL, pts_gl[2], pts_gl[1], X_gl, conv)
            tpse = translate(G, pts[2], pts[1], conv)
            tXse = translate_diff(G, pts[2], pts[1], X, conv)
            @test isapprox(G, tpse, project(SEGL, tpgl))
            @test isapprox(G, tpse, tXse, project(SEGL, tpgl, tXgl))

            @test isapprox(
                G,
                pts_gl[1],
                X_gl,
                translate_diff(G, Identity(G), pts_gl[1], X_gl, conv),
            )
        end
    end

    @testset "Adjoint action on 𝔰𝔢(3)" begin
        G = SpecialEuclidean(3)
        t = Vector{Float64}.([1:3, 2:4, 4:6])
        ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
        p = Matrix(I, 3, 3)
        Rn = Rotations(3)
        pts = [ProductRepr(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
        X = ProductRepr([-1.0, 2.0, 1.0], hat(Rn, p, [1.0, 0.5, -0.5]))
        q = ProductRepr([0.0, 0.0, 0.0], p)

        # adjoint action of SE(3)
        fX = TFVector(vee(G, q, X), VeeOrthogonalBasis())
        fXp = adjoint_action(G, pts[1], fX)
        fXp2 = adjoint_action(G, pts[1], X)
        @test isapprox(G, pts[1], hat(G, pts[1], fXp.data), fXp2)
    end
end
