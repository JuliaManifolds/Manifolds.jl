
include("../utils.jl")
include("group_utils.jl")

@testset "Rotation-translation action" begin
    G = SpecialEuclidean(2)
    M = base_manifold(G)
    A_left = RotationTranslationAction(Euclidean(2), G)
    A_right = RotationTranslationAction(Euclidean(2), G, RightAction())

    @test repr(A_left) ==
          "RotationTranslationAction($(repr(Euclidean(2))), $(repr(G)), LeftAction())"
    @test repr(A_right) ==
          "RotationTranslationAction($(repr(Euclidean(2))), $(repr(G)), RightAction())"

    types_a = [ArrayPartition{Float64,Tuple{Vector{Float64},Matrix{Float64}}}]

    types_m = [Vector{Float64}]

    @test group_manifold(A_left) == Euclidean(2)
    @test base_group(A_left) == G
    @test isa(A_left, AbstractGroupAction{LeftAction})
    @test base_manifold(G) == M

    for (i, T_A, T_M) in zip(1:length(types_a), types_a, types_m)
        angles = (0.0, π / 2, 2π / 3, π / 4)
        translations = [[1.0, 0.0], [0.0, -2.0], [-1.0, 2.0]]
        a_pts =
            convert.(
                T_A,
                [
                    ArrayPartition(t, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for
                    (t, ϕ) in zip(translations, angles)
                ],
            )
        a_X_pts = map(a -> log_lie(G, a), a_pts)
        m_pts = convert.(T_M, [[0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
        X_pts = convert.(T_M, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        atol = if eltype(T_M) == Float32
            2e-7
        else
            1e-15
        end
        test_action(
            A_left,
            a_pts,
            m_pts,
            X_pts;
            test_optimal_alignment=false,
            test_diff=true,
            atol=atol,
        )

        test_action(
            A_right,
            a_pts,
            m_pts,
            X_pts;
            test_optimal_alignment=false,
            test_diff=true,
            atol=atol,
        )

        @testset "apply_diff_group" begin
            @test apply_diff_group(A_left, Identity(G), a_X_pts[1], m_pts[1]) ≈
                  a_X_pts[1].x[2] * m_pts[1]
            Y = similar(m_pts[1])
            apply_diff_group!(A_left, Y, Identity(G), a_X_pts[1], m_pts[1])
            @test Y ≈ a_X_pts[1].x[2] * m_pts[1]
        end

        @testset "apply_diff" begin
            @test apply_diff(A_left, Identity(G), m_pts[1], X_pts[1]) ≈ X_pts[1]
            Y = similar(X_pts[1])
            apply_diff!(A_left, Y, Identity(G), m_pts[1], X_pts[1])
            @test Y ≈ X_pts[1]

            @test apply_diff(A_left, a_pts[1], m_pts[1], X_pts[1]) ≈
                  a_pts[1].x[2] * X_pts[1]
            Y = similar(X_pts[1])
            apply_diff!(A_left, Y, a_pts[1], m_pts[1], X_pts[1])
            @test Y ≈ a_pts[1].x[2] * X_pts[1]

            @test apply_diff(A_right, Identity(G), m_pts[1], X_pts[1]) ≈ X_pts[1]
            Y = similar(X_pts[1])
            apply_diff!(A_right, Y, Identity(G), m_pts[1], X_pts[1])
            @test Y ≈ X_pts[1]

            @test apply_diff(A_right, a_pts[1], m_pts[1], X_pts[1]) ≈
                  a_pts[1].x[2] \ X_pts[1]
            Y = similar(X_pts[1])
            apply_diff!(A_right, Y, a_pts[1], m_pts[1], X_pts[1])
            @test Y ≈ a_pts[1].x[2] \ X_pts[1]
        end
    end
end

@testset "Matrix columnwise multiplication action" begin
    M = Euclidean(2, 3)
    G = SpecialEuclidean(2)
    p1 = [
        0.4385117672460505 -0.6877826444042382 0.24927087715818771
        -0.3830259932279294 0.35347460720654283 0.029551386021386548
    ]
    p2 = [
        -0.42693314765896473 -0.3268567431952937 0.7537898908542584
        0.3054740561061169 -0.18962848284149897 -0.11584557326461796
    ]
    A = Manifolds.ColumnwiseSpecialEuclideanAction(M, G)

    @test group_manifold(A) === M
    @test base_group(A) === SpecialEuclidean(2)

    a1 = ArrayPartition(
        [1.0, 2.0],
        [0.5851302132737501 -0.8109393525500014; 0.8109393525500014 0.5851302132737504],
    )
    a2 = ArrayPartition(
        [2.0, -1.0],
        [0.903025374532402 -0.4295872122754759; 0.4295872122754759 0.9030253745324022],
    )
    a3 = ArrayPartition(
        [2.0, 0.0],
        [0.5851302132737501 -0.8109393525500014; 0.8109393525500014 0.5851302132737504],
    )
    @test isapprox(
        apply(A, a1, p1),
        [
            1.567197334849809 0.3109111254828243 1.1218915396673668
            2.1314863675092206 1.6490786599533187 2.2194349725374605
        ],
    )
    @test isapprox(
        inverse_apply(A, a1, p1),
        [
            -2.2610332854401007 -2.3228048546690494 -2.0371886150121097
            -0.9390476037204165 0.40525761065242144 -0.5441732289245019
        ],
    )
    @test apply(A, Identity(G), p1) === p1
    q = similar(p1)
    apply!(A, q, a1, p1)
    @test isapprox(q, apply(A, a1, p1))
    apply!(A, q, Identity(G), p1)
    @test isapprox(q, p1)
    test_action(
        A,
        [a1, a2, a3],
        [p1, p2];
        test_optimal_alignment=true,
        test_diff=false,
        test_switch_direction=false,
        atol=1e-14,
    )
end
