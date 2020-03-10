include("../utils.jl")
include("group_utils.jl")

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
        x = Matrix(I, n, n)

        if n == 2
            t = Vector{Float64}.([1:2, 2:3, 3:4])
            ω = [[1.0, 2.0], [2.0, 1.0], [1.0, 3.0]]
            tuple_pts = [(ti, exp(Rn, x, hat(Rn, x, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_v = ([-1.0, 2.0], hat(Rn, x, [1.0, -0.5]))
        elseif n == 3
            t = Vector{Float64}.([1:3, 2:4, 4:6])
            ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            tuple_pts = [(ti, exp(Rn, x, hat(Rn, x, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_v = ([-1.0, 2.0, 1.0], hat(Rn, x, [1.0, 0.5, -0.5]))
        else # n == 4
            t = Vector{Float64}.([1:4, 2:5, 3:6])
            ω = [
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                [1.0, 3.0, 2.0, 1.0, 2.0, 3.0],
            ]
            tuple_pts = [(ti, exp(Rn, x, hat(Rn, x, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_v = (
                [-1.0, 2.0, 1.0, 3.0],
                hat(Rn, x, [1.0, 0.5, -0.5, 0.0, 2.0, 1.0]),
            )
        end

        @testset "product point" begin
            reshapers = (Manifolds.ArrayReshaper(), Manifolds.StaticReshaper())
            for reshaper in reshapers
                shape_se = Manifolds.ShapeSpecification(reshaper, M.manifolds...)
                pts = [Manifolds.prod_point(shape_se, tp...) for tp in tuple_pts]
                v_pts = [Manifolds.prod_point(shape_se, tuple_v...)]

                g1, g2 = pts[1:2]
                t1, R1 = g1.parts
                t2, R2 = g2.parts
                g1g2 = Manifolds.prod_point(shape_se, R1 * t2 + t1, R1 * R2)
                @test isapprox(G, compose(G, g1, g2), g1g2)
                @test affine_matrix(G, g1g2) ≈ affine_matrix(G, g1) * affine_matrix(G, g2)
                tmp = copy(g1)
                Manifolds._padpoint!(G, tmp)
                @test tmp == g1
                tmp = copy(v_pts[1])
                Manifolds._padvector!(G, tmp)
                @test tmp == v_pts[1]

                w = translate_diff(G, pts[1], make_identity(G, pts[1]), v_pts[1])
                w2 = allocate(w)
                w2.parts[1] .= w.parts[1]
                w2.parts[2] .= pts[1].parts[2] * w.parts[2]
                @test screw_matrix(G, w2) ≈ affine_matrix(G, pts[1]) * screw_matrix(G, v_pts[1])

                test_group(G, pts, v_pts, v_pts; test_diff = true, diff_convs = [(), (LeftAction(),)])
            end
        end

        @testset "product repr" begin
            pts = [ProductRepr(tp...) for tp in tuple_pts]
            v_pts = [ProductRepr(tuple_v...)]

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

            w = translate_diff(G, pts[1], make_identity(G, pts[1]), v_pts[1])
            w2 = allocate(w)
            w2.parts[1] .= w.parts[1]
            w2.parts[2] .= pts[1].parts[2] * w.parts[2]
            w2mat = screw_matrix(G, w2)
            @test w2mat ≈ affine_matrix(G, pts[1]) * screw_matrix(G, v_pts[1])
            @test screw_matrix(G, w2mat) === w2mat

            test_group(G, pts, v_pts, v_pts; test_diff = true, diff_convs = [(), (LeftAction(),)])
        end

        @testset "affine matrix" begin
            pts = [affine_matrix(G, ProductRepr(tp...)) for tp in tuple_pts]
            v_pts = [screw_matrix(G, ProductRepr(tuple_v...))]
            test_group(G, pts, v_pts, v_pts; test_diff = true, diff_convs = [(), (LeftAction(),)])
        end

        @testset "hat/vee" begin
            shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M.manifolds...)
            x = Manifolds.prod_point(shape_se, tuple_pts[1]...)
            V = Manifolds.prod_point(shape_se, tuple_v...)
            vexp = [V.parts[1]; vee(Rn, x.parts[2], V.parts[2])]
            v = vee(G, x, V)
            @test v ≈ vexp
            @test hat(G, x, v) ≈ V

            v = vee(G, affine_matrix(G, x), screw_matrix(G, V))
            @test v ≈ vexp
            @test hat(G, affine_matrix(G, x), v) ≈ screw_matrix(G, V)
        end
    end

    G = SpecialEuclidean(11)
    @test affine_matrix(G, make_identity(G, ones(12, 12))) isa Diagonal{Float64,Vector{Float64}}
    @test affine_matrix(G, make_identity(G, ones(12, 12))) == Diagonal(ones(11))
end
