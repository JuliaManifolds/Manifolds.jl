
include("../utils.jl")
include("group_utils.jl")

@testset "Rotation action" begin
    M = Rotations(2)
    G = SpecialOrthogonal(2)
    A_left = RotationAction(Euclidean(2), G)
    A_right = RotationAction(Euclidean(2), G, RightAction())

    @test repr(A_left) == "RotationAction($(repr(Euclidean(2))), $(repr(G)), LeftAction())"
    @test repr(A_right) ==
          "RotationAction($(repr(Euclidean(2))), $(repr(G)), RightAction())"

    types_a = [Matrix{Float64}]

    types_m = [Vector{Float64}]

    @test group_manifold(A_left) == Euclidean(2)
    @test base_group(A_left) == G
    @test isa(A_left, AbstractGroupAction{LeftAction})
    @test base_manifold(G) == M

    for (i, T_A, T_M) in zip(1:length(types_a), types_a, types_m)
        angles = (0.0, π / 2, 2π / 3, π / 4)
        a_pts = convert.(T_A, [[cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)] for ϕ in angles])
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
            test_optimal_alignment=true,
            test_diff=true,
            atol=atol,
        )

        test_action(
            A_right,
            a_pts,
            m_pts,
            X_pts;
            test_optimal_alignment=true,
            test_diff=true,
            atol=atol,
        )
    end
end

@testset "Rotation around axis action" begin
    M = Circle()
    G = RealCircleGroup()
    axis = [sqrt(2) / 2, sqrt(2) / 2, 0.0]
    A = Manifolds.RotationAroundAxisAction(axis)

    types_a = [Ref(Float64)]

    types_m = [Vector{Float64}]

    @test group_manifold(A) == Euclidean(3)
    @test base_group(A) == G
    @test isa(A, AbstractGroupAction{LeftAction})
    @test base_manifold(G) == M

    for (i, T_A, T_M) in zip(1:length(types_a), types_a, types_m)
        angles = (0.0, π / 2, 2π / 3, π / 4)
        a_pts = convert.(T_A, [angles...])
        m_pts = convert.(T_M, [[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [1.0, 1.0, -2.0]])
        X_pts = convert.(T_M, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        atol = if eltype(T_M) == Float32
            2e-7
        else
            1e-15
        end
        test_action(
            A,
            a_pts,
            m_pts,
            X_pts;
            test_optimal_alignment=false,
            test_diff=false,
            test_mutating_group=false,
            test_switch_direction=false,
            atol=atol,
        )
    end
end

@testset "Matrix columnwise multiplication action" begin
    M = Stiefel(4, 2)
    G = Orthogonal(2)
    A = Manifolds.RowwiseMultiplicationAction(M, G)

    @test group_manifold(A) === M
    @test base_group(A) === G

    p = [
        -0.6811974718784779 0.6307712613923999
        -0.3223041997065754 -0.6105469895916442
        0.4756685377015621 0.47878051009766454
        -0.45368430587659947 -0.011367165550016604
    ]
    a = [0.5851302132737501 -0.8109393525500014; 0.8109393525500014 0.5851302132737504]
    @test isapprox(
        apply(A, a, p),
        [
            -0.9101064603224936 -0.1833265140983431
            0.30652665532746654 -0.6186186492676017
            -0.10993392395923157 0.6658872779768721
            -0.25624631278506876 -0.3745617292722656
        ],
    )
    @test isapprox(
        inverse_apply(A, a, p),
        [
            0.11292801631890707 0.9214931595093182
            -0.6837065055541344 -0.09588033119920833
            0.6665899897850875 -0.10558939400734933
            -0.2746824765279873 0.3612591852670672
        ],
    )
    @test apply(A, Identity(G), p) === p
    q = similar(p)
    apply!(A, q, a, p)
    @test isapprox(q, apply(A, a, p))
end
