include("../utils.jl")
include("group_utils.jl")

function _to_affine(p, is_vector = false)
    d = length(p)
    N = Int((sqrt(4d + 1) - 1) / 2)
    P = Matrix{eltype(p)}(undef, N + 1, N + 1)
    setindex!(P, p.parts[1], 1:N, N + 1)
    setindex!(P, p.parts[2], 1:N, 1:N)
    fill!(view(P, N + 1, 1:N), 0)
    if is_vector
        P[N+1, N+1] = 0
    else
        P[N+1, N+1] = 1
    end
    return P
end

@testset "Special Euclidean group" begin
    @testset "SpecialEuclidean($n)" for n in (2, 3, 4)
        G = SpecialEuclidean(n)
        @test repr(G) == "SpecialEuclidean($n)"
        M = base_manifold(G)
        @test M === TranslationGroup(n) × SpecialOrthogonal(n)
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
                @test _to_affine(g1g2) ≈ _to_affine(g1) * _to_affine(g2)

                w = translate_diff(G, pts[1], Identity(G), v_pts[1])
                w2 = similar(w)
                w2.parts[1] .= w.parts[1]
                w2.parts[2] .= pts[1].parts[2] * w.parts[2]
                @test _to_affine(w2, true) ≈ _to_affine(pts[1]) * _to_affine(v_pts[1], true)

                test_group(G, pts, v_pts, v_pts; test_diff = true, diff_convs = [(), (LeftAction(),)])
            end
        end

        @testset "affine matrix" begin
            shape_se = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), M.manifolds...)
            pts = [_to_affine(Manifolds.prod_point(shape_se, tp...)) for tp in tuple_pts]
            v_pts = [_to_affine(Manifolds.prod_point(shape_se, tuple_v...), true)]
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

            v = vee(G, _to_affine(x), _to_affine(V, true))
            @test v ≈ vexp
            @test hat(G, _to_affine(x), v) ≈ _to_affine(V, true)
        end
    end
end
