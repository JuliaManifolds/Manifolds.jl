include("../utils.jl")
include("group_utils.jl")

using ManifoldsBase: VeeOrthogonalBasis

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
            tuple_X = ([-1.0, 2.0], hat(Rn, p, [1.0]))
        elseif n == 3
            t = Vector{Float64}.([1:3, 2:4, 4:6])
            ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = ([-1.0, 2.0, 1.0], hat(Rn, p, [1.0, 0.5, -0.5]))
        else # n == 4
            t = Vector{Float64}.([1:4, 2:5, 3:6])
            ω = [
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                [1.0, 3.0, 2.0, 1.0, 2.0, 3.0],
            ]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = ([-1.0, 2.0, 1.0, 3.0], hat(Rn, p, [1.0, 0.5, -0.5, 0.0, 2.0, 1.0]))
        end

        @testset "product point" begin
            reshapers = (Manifolds.ArrayReshaper(), Manifolds.StaticReshaper())
            for reshaper in reshapers
                shape_se = Manifolds.ShapeSpecification(reshaper, M.manifolds...)
                pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]
                X_pts = [Manifolds.prod_point(shape_se, tuple_X...)]

                g1, g2 = pts[1:2]
                t1, R1 = g1.parts
                t2, R2 = g2.parts
                g1g2 = Manifolds.prod_point(shape_se, R1 * t2 + t1, R1 * R2)
                @test isapprox(G, compose(G, g1, g2), g1g2)
                @test affine_matrix(G, g1g2) ≈ affine_matrix(G, g1) * affine_matrix(G, g2)
                tmp = copy(g1)
                Manifolds._padpoint!(G, tmp)
                @test tmp == g1
                tmp = copy(X_pts[1])
                Manifolds._padvector!(G, tmp)
                @test tmp == X_pts[1]

                w = translate_diff(G, pts[1], make_identity(G, pts[1]), X_pts[1])
                w2 = allocate(w)
                w2.parts[1] .= w.parts[1]
                w2.parts[2] .= pts[1].parts[2] * w.parts[2]
                @test screw_matrix(G, w2) ≈
                      affine_matrix(G, pts[1]) * screw_matrix(G, X_pts[1])

                test_group(
                    G,
                    pts,
                    X_pts,
                    X_pts;
                    test_diff=true,
                    diff_convs=[(), (LeftAction(),), (RightAction(),)],
                )
            end
        end

        @testset "product repr" begin
            pts = [ProductRepr(tp...) for tp in tuple_pts]
            X_pts = [ProductRepr(tuple_X...)]

            g1, g2 = pts[1:2]
            t1, R1 = g1.parts
            t2, R2 = g2.parts
            g1g2 = ProductRepr(R1 * t2 + t1, R1 * R2)
            @test isapprox(G, compose(G, g1, g2), g1g2)
            g1g2mat = affine_matrix(G, g1g2)
            @test g1g2mat ≈ affine_matrix(G, g1) * affine_matrix(G, g2)
            @test affine_matrix(G, g1g2mat) === g1g2mat
            @test affine_matrix(G, make_identity(G, pts[1])) isa SDiagonal{n,Float64}
            @test affine_matrix(G, make_identity(G, pts[1])) == SDiagonal{n,Float64}(I)

            w = translate_diff(G, pts[1], make_identity(G, pts[1]), X_pts[1])
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
                diff_convs=[(), (LeftAction(),), (RightAction(),)],
            )
        end

        @testset "affine matrix" begin
            pts = [affine_matrix(G, ProductRepr(tp...)) for tp in tuple_pts]
            X_pts = [screw_matrix(G, ProductRepr(tuple_X...))]
            test_group(
                G,
                pts,
                X_pts,
                X_pts;
                test_diff=true,
                diff_convs=[(), (LeftAction(),), (RightAction(),)],
            )
        end

        @testset "hat/vee" begin
            shape_se =
                Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M.manifolds...)
            p = Manifolds.prod_point(shape_se, tuple_pts[1]...)
            V = Manifolds.prod_point(shape_se, tuple_X...)
            vexp = [V.parts[1]; vee(Rn, p.parts[2], V.parts[2])]
            v = vee(G, p, V)
            @test v ≈ vexp
            @test hat(G, p, v) ≈ V

            v = vee(G, affine_matrix(G, p), screw_matrix(G, V))
            @test v ≈ vexp
            @test hat(G, affine_matrix(G, p), v) ≈ screw_matrix(G, V)
        end
    end

    G = SpecialEuclidean(11)
    @test affine_matrix(G, make_identity(G, ones(12, 12))) isa
          Diagonal{Float64,Vector{Float64}}
    @test affine_matrix(G, make_identity(G, ones(12, 12))) == Diagonal(ones(11))

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
        fXp2 = vee(G, q, compose(G, pts[1], exp(G, q, X)))
    end
end
