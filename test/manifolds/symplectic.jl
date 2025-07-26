include("../header.jl")
using FiniteDifferences
using Manifolds: RiemannianProjectionBackend
using ManifoldDiff

@testset "SymplecticMatrices" begin
    M = SymplecticMatrices(2)
    Metr_Sp_2 = MetricManifold(M, RealSymplecticMetric())

    p_2 = [0.0 1.0 / 2.0; -2.0 -2.0]
    X1 = [
        -0.121212 0.121212
        0.969697 -1.0
    ]
    X2 = [
        0.0 0.0
        0.0 -1.0
    ]

    Sp_6 = SymplecticMatrices(6)
    points = [
        [
            1.0 1.0 3.0 0.0 0.0 0.0
            -1.0 -2.0 -4.0 0.0 0.0 0.0
            3.0 0.0 7.0 0.0 0.0 0.0
            3.0 4.0 10.0 14.0 5.0 -6.0
            -5.0 -9.0 -19.0 7.0 2.0 -3.0
            0.0 0.0 0.0 -2.0 -1.0 1.0
        ],
        [
            0.0 0.0 0.0 -2.0 1.0 5.0
            0.0 0.0 0.0 0.0 -1.0 -2.0
            0.0 0.0 0.0 -1.0 -2.0 -3.0
            -1.0 2.0 -1.0 9.0 -4.0 -21.0
            -7.0 11.0 -5.0 -2.0 10.0 24.0
            3.0 -4.0 2.0 -2.0 -7.0 -13.0
        ],
        [
            6.0 -1.0 11.0 -10.0 8.0 6.0
            2.0 0.0 3.0 23.0 -14.0 -14.0
            -3.0 0.0 -5.0 -1.0 5.0 1.0
            0.0 0.0 0.0 0.0 -1.0 0.0
            0.0 0.0 0.0 5.0 -3.0 -3.0
            0.0 0.0 0.0 3.0 -4.0 -2.0
        ],
        [
            2.0 -1.0 5.0 -4.0 4.0 2.0
            2.0 0.0 3.0 11.0 -6.0 -6.0
            -1.0 0.0 -2.0 -4.0 7.0 3.0
            0.0 0.0 0.0 0.0 -1.0 0.0
            0.0 0.0 0.0 2.0 -1.0 -1.0
            0.0 0.0 0.0 3.0 -4.0 -2.0
        ],
    ]

    large_tr_norm_points = [
        [
            0.0 -3.0 -5.0 0.0 0.0 0.0
            -2.0 -3.0 4.0 0.0 0.0 0.0
            -3.0 -5.0 5.0 0.0 0.0 0.0
            11.0 27.0 -5.0 5.0 -2.0 1.0
            -37.0 -29.0 117.0 40.0 -15.0 9.0
            22.0 30.0 -47.0 -27.0 10.0 -6.0
        ],
        [
            0.0 0.0 0.0 7.0 -2.0 1.0
            0.0 0.0 0.0 19.0 -5.0 3.0
            0.0 0.0 0.0 -6.0 2.0 -1.0
            -1.0 1.0 8.0 -14.0 4.0 -2.0
            0.0 -1.0 -2.0 -94.0 26.0 -15.0
            -1.0 -2.0 3.0 45.0 -11.0 7.0
        ],
    ]

    @testset "Basics" begin
        @test repr(M) == "SymplecticMatrices($(2), ℝ)"
        @test_throws ArgumentError SymplecticMatrices(3)
        @test representation_size(M) == (2, 2)
        @test !is_flat(M)

        @test is_point(M, p_2)
        @test_throws DomainError is_point(M, p_2 + I; error = :error)

        @test is_vector(M, p_2, X1; atol = 1.0e-6)
        @test is_vector(M, p_2, X2; atol = 1.0e-12)
        @test is_vector(M, p_2, X1 + X2; atol = 1.0e-6)
        @test_throws DomainError is_vector(M, p_2, X1 + [0.1 0.1; -0.1 0.1]; error = :error)
    end
    @testset "Symplectic Inverse" begin
        I_2n = Array(I, 2, 2)
        @test Manifolds.symplectic_inverse_times(M, p_2, p_2) == I_2n
        @test Manifolds.symplectic_inverse_times!(M, copy(p_2), p_2, p_2) == I_2n
        @test inv(M, p_2) * p_2 == I_2n
        @test inv!(M, copy(p_2)) * p_2 == I_2n
    end
    @testset "Embedding" begin
        x = [0.0 1.0 / 2.0; -2.0 -2.0]
        y = similar(x)
        z = embed(M, x)
        @test z == x

        Y = similar(X1)
        embed!(M, Y, p_2, X1)
        @test Y == X1
    end
    @testset "Retractions and Exponential Mapping" begin
        q_exp = [
            -0.0203171 0.558648
            -1.6739 -3.19344
        ]
        @test isapprox(exp(M, p_2, X2), q_exp; atol = 1.0e-5)
        @test isapprox(retract(M, p_2, X2, ExponentialRetraction()), q_exp; atol = 1.0e-5)

        q_cay = [
            0.0 0.5
            -2.0 -3.0
        ]
        @test retract(M, p_2, X2) == q_cay
        @test retract(M, p_2, X2, CayleyRetraction()) == q_cay

        X_inv_cayley_retraction = inverse_retract(M, p_2, q_cay)
        X_inv_cayley_retraction_2 =
            inverse_retract(M, p_2, q_cay, CayleyInverseRetraction())
        @test X_inv_cayley_retraction == X_inv_cayley_retraction_2
        @test X_inv_cayley_retraction ≈ X2
    end
    @testset "Riemannian metric" begin
        X1_p_norm = 0.49259905148939337
        @test norm(M, p_2, X1) == X1_p_norm
        @test norm(M, p_2, X1) == √(inner(M, p_2, X1, X1))

        X2_p_norm = 1 / 2
        @test norm(M, p_2, X2) == X2_p_norm
        @test norm(M, p_2, X2) == √(inner(M, p_2, X2, X2))

        q_2 = retract(M, p_2, X2, ExponentialRetraction())
        approximate_p_q_geodesic_distance = 0.510564444555605
        @test isapprox(distance(M, p_2, q_2), approximate_p_q_geodesic_distance; atol = 1.0e-14)

        # Project tangent vector into (T_pSp)^{\perp}:
        Extended_Sp_2 = MetricManifold(get_embedding(M), ExtendedSymplecticMetric())
        proj_normal_X2 = Manifolds.project_normal!(Extended_Sp_2, copy(X2), p_2, X2)
        @test isapprox(proj_normal_X2, zero(X2); atol = 1.0e-16)

        # Project Project matrix A ∈ ℝ^{2×2} onto (T_pSp):
        A_2 = [5.0 -21.5; 3.14 14.9]
        A_2_proj = similar(A_2)
        project!(Extended_Sp_2, A_2_proj, p_2, A_2)
        @test is_vector(M, p_2, A_2_proj; atol = 1.0e-16)

        # Change representer of A onto T_pSp:
        @testset "Change Representer" begin
            A_2_representer = change_representer(
                MetricManifold(get_embedding(M), ExtendedSymplecticMetric()),
                EuclideanMetric(),
                p_2,
                A_2,
            )
            @test isapprox(inner(M, p_2, A_2_representer, X1), tr(A_2' * X1); atol = 1.0e-12)
            @test isapprox(inner(M, p_2, A_2_representer, X2), tr(A_2' * X2); atol = 1.0e-12)
            @test isapprox(inner(M, p_2, A_2_representer, A_2), norm(A_2)^2; atol = 1.0e-12)
        end
    end
    @testset "Generate random points/tangent vectors" begin
        M_big = SymplecticMatrices(20)
        Random.seed!(49)
        p_big = rand(M_big)
        @test is_point(M_big, p_big; error = :error, atol = 1.0e-9)
        X_big = rand(M_big; vector_at = p_big)
        @test is_vector(M_big, p_big, X_big; error = :error, atol = 1.0e-9)
    end
    @testset "test_manifold(SymplecticMatrices(6))" begin
        test_manifold(
            Sp_6,
            cat(points, large_tr_norm_points; dims = 1);
            retraction_methods = [CayleyRetraction(), ExponentialRetraction()],
            default_retraction_method = CayleyRetraction(),
            default_inverse_retraction_method = CayleyInverseRetraction(),
            test_inplace = true,
            is_point_atol_multiplier = 1.0e8,
            is_tangent_atol_multiplier = 1.0e6,
            retraction_atol_multiplier = 1.0e4,
            test_project_tangent = true,
            test_injectivity_radius = false,
            test_exp_log = false,
            test_representation_size = true,
        )
    end
    @testset "Gradient Computations" begin
        test_f(p) = tr(p)
        J = SymplecticElement(points[1])
        analytical_grad_f(p) = (1 / 2) * (p * J * p * J + p * p')

        p_grad = points[1]
        fd_diff = RiemannianProjectionBackend(AutoFiniteDifferences(central_fdm(5, 1)))

        @test isapprox(
            Manifolds.gradient(Sp_6, test_f, p_grad, fd_diff),
            analytical_grad_f(p_grad);
            atol = 1.0e-9,
        )
        @test isapprox(
            Manifolds.gradient(Sp_6, test_f, p_grad, fd_diff; extended_metric = false),
            analytical_grad_f(p_grad);
            atol = 1.0e-9,
        )

        grad_f_p = similar(p_grad)
        Manifolds.gradient!(Sp_6, test_f, grad_f_p, p_grad, fd_diff)
        @test isapprox(grad_f_p, analytical_grad_f(p_grad); atol = 1.0e-9)

        Manifolds.gradient!(Sp_6, test_f, grad_f_p, p_grad, fd_diff; extended_metric = false)
        @test isapprox(grad_f_p, analytical_grad_f(p_grad); atol = 1.0e-9)

        X = riemannian_gradient(Sp_6, p_grad, one(p_grad))
        X2 = similar(X)
        riemannian_gradient!(Sp_6, X2, p_grad, one(p_grad))
        @test isapprox(Sp_6, p_grad, X, X2)
    end
    @testset "SymplecticElement" begin
        @test SymplecticElement() == SymplecticElement(1)
        Sp_4 = SymplecticMatrices(4)
        pQ_1 = [
            0.0 0 -2.0 3.0
            0.0 0.0 1.0 -1.0
            -1.0 -1.0 0.0 0.0
            -3.0 -2.0 4.0 -4.0
        ]
        pQ_2 = [
            0.0 0.0 -2.0 5.0
            0.0 0.0 1.0 -2.0
            -2.0 -1.0 0.0 0.0
            -5.0 -2.0 4 -8.0
        ]
        p_odd_row = [
            0.0 0.0 -1.0 3.0
            0.0 0.0 1.0 0.0
            -2.0 -1.0 0.0 0.0
        ]
        p_odd_col = [
            0.0 -2.0 5.0
            0.0 1.0 -2.0
            -1.0 0.0 0.0
            -2.0 4.0 -8.0
        ]
        Q = SymplecticElement(pQ_1, pQ_2)
        Q2 = SymplecticElement(1.0)

        @testset "Type Basics" begin
            @test Q == Q2
            @test ndims(Q) == 2
            @test copy(Q) == Q
            @test eltype(SymplecticElement(1 // 1)) == Rational{Int64}
            @test convert(SymplecticElement{Float64}, Q) == SymplecticElement(1.0)
            @test "$Q" == "SymplecticElement{Float64}(): 1.0*[0 I; -I 0]"
            @test (
                "$(SymplecticElement(1 + 2im))" ==
                    "SymplecticElement{Complex{Int64}}(): (1 + 2im)*[0 I; -I 0]"
            )
        end

        @testset "Matrix Operations" begin
            @test -Q == SymplecticElement(-1.0)
            @test (2 * Q) * (5 / 6) == SymplecticElement(5 / 3)

            @testset "Powers" begin
                @test inv(Q) * Q == I
                @test (
                    inv(SymplecticElement(-4.0 + 8im)) * SymplecticElement(-4.0 + 8im) ==
                        UniformScaling(1.0 + 0.0im)
                )
                @test Q * Q == -I
                @test Q^2 == -I
                @test Q^3 == -Q
                @test Q^4 == I
            end
            @testset "Addition (subtraction)" begin
                @test Q + Q == 2 * Q
                @test Q - SymplecticElement(1.0) == SymplecticElement(0.0)
                @test Q + pQ_1 == [
                    0.0 0.0 -1.0 3.0
                    0.0 0.0 1.0 0.0
                    -2.0 -1.0 0.0 0.0
                    -3.0 -3.0 4.0 -4.0
                ]
                @test Q - pQ_1 == [
                    0.0 0.0 3.0 -3.0
                    0.0 0.0 -1.0 2.0
                    0.0 1.0 0.0 0.0
                    3.0 1.0 -4.0 4.0
                ]
                @test pQ_1 - Q == [
                    0.0 0.0 -3.0 3.0
                    0.0 0.0 1.0 -2.0
                    0.0 -1.0 0.0 0.0
                    -3.0 -1.0 4.0 -4.0
                ]
                @test (pQ_1 + Q) == (Q + pQ_1)

                @test_throws ArgumentError Q + p_odd_row
            end
            @testset "Transpose-Adjoint" begin
                @test Q' == SymplecticElement(-1.0)
                @test transpose(SymplecticElement(10.0)) == SymplecticElement(-10.0)
                @test transpose(SymplecticElement(1 - 2.0im)) ==
                    SymplecticElement(-1 + 2.0im)
                @test adjoint(Q) == -Q
                @test adjoint(SymplecticElement(1 - 2.0im)) == SymplecticElement(-1 - 2.0im)
                @test adjoint(SymplecticElement(-1im)) == SymplecticElement(-1im)
                @test adjoint(SymplecticElement(2.0)) == SymplecticElement(-2.0)
            end
            @testset "Inplace mul!" begin
                z1 = [1 + 2im; 1 - 2im]
                @test lmul!(Q, copy(z1)) == Q * z1
                @test lmul!(Q, copy(p_odd_col)) == [
                    -1 0 0.0
                    -2 4 -8.0
                    0 2 -5.0
                    0 -1 2.0
                ]
                @test_throws ArgumentError lmul!(Q, copy(p_odd_row))

                @test rmul!(copy(z1'), Q) == z1' * Q
                @test rmul!(copy(p_odd_row), Q) == [
                    1 -3 0 0.0
                    -1 0 0 0.0
                    0 0 -2 -1.0
                ]
                @test_throws ArgumentError rmul!(copy(p_odd_col), Q)

                z1_copy = copy(z1)
                mul!(z1_copy, Q, z1)
                @test z1_copy == Q * z1

                @test_throws ArgumentError mul!(copy(p_odd_row), Q, p_odd_row)
                @test_throws ArgumentError mul!(copy(p_odd_row), p_odd_col, Q)

                @test_throws ArgumentError Q * p_odd_row
                @test_throws ArgumentError p_odd_col * Q

                mul!(z1_copy', z1', Q)
                @test z1_copy' == (z1' * Q)
            end
        end

        @testset "Symplectic Inverse Ops." begin
            @test ((Q' * pQ_1' * Q) * pQ_1 - I) == zeros(eltype(pQ_1), size(pQ_1)...)
        end
    end
    @testset "field parameter" begin
        M = SymplecticMatrices(2; parameter = :field)
        @test typeof(get_embedding(M)) === Euclidean{Tuple{Int, Int}, ℝ}
        @test repr(M) == "SymplecticMatrices(2, ℝ; parameter=:field)"
    end
end
