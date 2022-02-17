include("../utils.jl")

@testset "Symplectic" begin
    @testset "Real" begin
        Sp_2 = Symplectic(2 * 1)
        Metr_Sp_2 = MetricManifold(Sp_2, RealSymplecticMetric())

        p_2 = [0.0 1.0/2.0; -2.0 -2.0]
        X1 = [
            -0.121212 0.121212
            0.969697 -1.0
        ]
        X2 = [
            0.0 0.0
            0.0 -1.0
        ]

        Sp_6 = Symplectic(6)
        points = [
            [
                1 1 3 0 0 0
                -1 -2 -4 0 0 0
                3 0 7 0 0 0
                3 4 10 14 5 -6
                -5 -9 -19 7 2 -3
                0 0 0 -2 -1 1
            ],
            [
                0 0 0 -2 1 5
                0 0 0 0 -1 -2
                0 0 0 -1 -2 -3
                -1 2 -1 9 -4 -21
                -7 11 -5 -2 10 24
                3 -4 2 -2 -7 -13
            ],
            [
                6 -1 11 -10 8 6
                2 0 3 23 -14 -14
                -3 0 -5 -1 5 1
                0 0 0 0 -1 0
                0 0 0 5 -3 -3
                0 0 0 3 -4 -2
            ],
            [
                2 -1 5 -4 4 2
                2 0 3 11 -6 -6
                -1 0 -2 -4 7 3
                0 0 0 0 -1 0
                0 0 0 2 -1 -1
                0 0 0 3 -4 -2
            ],
        ]

        large_tr_norm_points = [
            [
                0 -3 -5 0 0 0
                -2 -3 4 0 0 0
                -3 -5 5 0 0 0
                11 27 -5 5 -2 1
                -37 -29 117 40 -15 9
                22 30 -47 -27 10 -6
            ],
            [
                0 0 0 7 -2 1
                0 0 0 19 -5 3
                0 0 0 -6 2 -1
                -1 1 8 -14 4 -2
                0 -1 -2 -94 26 -15
                -1 -2 3 45 -11 7
            ],
        ]

        @testset "Basics" begin
            @test repr(Sp_2) == "Symplectic{$(2), ℝ}()"
            @test representation_size(Sp_2) == (2, 2)
            @test base_manifold(Sp_2) === Sp_2
            @test (@inferred Manifolds.default_metric_dispatch(Metr_Sp_2)) === Val(true)

            @test is_point(Sp_2, p_2)
            @test_throws DomainError is_point(Sp_2, p_2 + I, true)

            @test is_vector(Sp_2, p_2, X1; atol=1.0e-6)
            @test is_vector(Sp_2, p_2, X2; atol=1.0e-12)
            @test is_vector(Sp_2, p_2, X1 + X2; atol=1.0e-6)
            @test_throws DomainError is_vector(Sp_2, p_2, X1 + [0.1 0.1; -0.1 0.1], true)
        end
        @testset "Symplectic Inverse" begin
            I_2n = Array(I, 2, 2)
            @test Manifolds.symplectic_inverse_times(Sp_2, p_2, p_2) == I_2n
            @test Manifolds.symplectic_inverse_times!(Sp_2, copy(p_2), p_2, p_2) == I_2n
            @test inv(Sp_2, p_2) * p_2 == I_2n
            @test inv!(Sp_2, copy(p_2)) * p_2 == I_2n
        end
        @testset "Embedding" begin
            x = [0.0 1.0/2.0; -2.0 -2.0]
            y = similar(x)
            z = embed(Sp_2, x)
            @test z == x

            Y = similar(X1)
            embed!(Sp_2, Y, p_2, X1)
            @test Y == X1
        end
        @testset "Retractions and Exponential Mapping" begin
            q_exp = [
                -0.0203171 0.558648
                -1.6739 -3.19344
            ]
            @test isapprox(exp(Sp_2, p_2, X2), q_exp; atol=1.0e-5)
            @test isapprox(
                retract(Sp_2, p_2, X2, ExponentialRetraction()),
                q_exp;
                atol=1.0e-5,
            )

            q_cay = [
                0.0 0.5
                -2.0 -3.0
            ]
            @test retract(Sp_2, p_2, X2) == q_cay
            @test retract(Sp_2, p_2, X2, CayleyRetraction()) == q_cay

            X_inv_cayley_retraction = inverse_retract(Sp_2, p_2, q_cay)
            X_inv_cayley_retraction_2 =
                inverse_retract(Sp_2, p_2, q_cay, CayleyInverseRetraction())
            @test X_inv_cayley_retraction == X_inv_cayley_retraction_2
            @test X_inv_cayley_retraction ≈ X2
        end
        @testset "Riemannian metric" begin
            X1_p_norm = 0.49259905148939337
            @test norm(Sp_2, p_2, X1) == X1_p_norm
            @test norm(Sp_2, p_2, X1) == √(inner(Sp_2, p_2, X1, X1))

            X2_p_norm = 1 / 2
            @test norm(Sp_2, p_2, X2) == X2_p_norm
            @test norm(Sp_2, p_2, X2) == √(inner(Sp_2, p_2, X2, X2))

            q_2 = retract(Sp_2, p_2, X2, ExponentialRetraction())
            approximate_p_q_geodesic_distance = 0.510564444555605
            @test isapprox(
                distance(Sp_2, p_2, q_2),
                approximate_p_q_geodesic_distance;
                atol=1.0e-16,
            )

            # Project tangent vector into (T_pSp)^{\perp}:
            Extended_Sp_2 = MetricManifold(get_embedding(Sp_2), ExtendedSymplecticMetric())
            proj_normal_X2 = Manifolds.project_normal!(Extended_Sp_2, copy(X2), p_2, X2)
            @test isapprox(proj_normal_X2, zero(X2); atol=1.0e-16)

            # Project Project matrix A ∈ ℝ^{2 × 2} onto (T_pSp):
            A_2 = [5.0 -21.5; 3.14 14.9]
            A_2_proj = similar(A_2)
            Manifolds.project!(Extended_Sp_2, A_2_proj, p_2, A_2)
            @test is_vector(Sp_2, p_2, A_2_proj; atol=1.0e-16)

            # Change representer of A onto T_pSp:
            @testset "Change Representer" begin
                A_2_representer = change_representer(
                    MetricManifold(get_embedding(Sp_2), ExtendedSymplecticMetric()),
                    EuclideanMetric(),
                    p_2,
                    A_2,
                )
                @test isapprox(
                    inner(Sp_2, p_2, A_2_representer, X1),
                    tr(A_2' * X1);
                    atol=1.0e-12,
                )
                @test isapprox(
                    inner(Sp_2, p_2, A_2_representer, X2),
                    tr(A_2' * X2);
                    atol=1.0e-12,
                )
                @test isapprox(
                    inner(Sp_2, p_2, A_2_representer, A_2),
                    norm(A_2)^2;
                    atol=1.0e-12,
                )
            end
        end
        @testset "Generate random points/tangent vectors" begin
            M_big = Symplectic(20)
            p_big = rand(M_big)
            @test is_point(M_big, p_big, true; atol=1.0e-12)
            X_big = rand(M_big; vector_at=p_big)
            @test is_vector(M_big, p_big, X_big, true; atol=1.0e-12)
        end
        @testset "test_manifold(Symplectic(6), ...)" begin
            @testset "Type $(Matrix{Float64})" begin
                type = Matrix{Float64}
                pts = convert.(type, points)
                test_manifold(
                    Sp_6,
                    cat(pts, large_tr_norm_points; dims=1);
                    retraction_methods=[CayleyRetraction(), ExponentialRetraction()],
                    default_retraction_method=CayleyRetraction(),
                    default_inverse_retraction_method=CayleyInverseRetraction(),
                    test_inplace=true,
                    is_point_atol_multiplier=1.0e8,
                    is_tangent_atol_multiplier=1.0e6,
                    retraction_atol_multiplier=1.0e4,
                    test_reverse_diff=false,
                    test_forward_diff=false,
                    test_project_tangent=true,
                    test_injectivity_radius=false,
                    test_exp_log=false,
                    test_representation_size=true,
                )
            end

            TEST_FLOAT32 && @testset "Type $(Matrix{Float32})" begin
                type = Matrix{Float64}
                pts = convert.(type, points)
                test_manifold(
                    Sp_6,
                    pts;
                    retraction_methods=[CayleyRetraction(), ExponentialRetraction()],
                    default_retraction_method=CayleyRetraction(),
                    default_inverse_retraction_method=CayleyInverseRetraction(),
                    test_inplace=true,
                    is_point_atol_multiplier=1.0e8,
                    is_tangent_atol_multiplier=1.0e6,
                    retraction_atol_multiplier=1.0e4,
                    test_reverse_diff=false,
                    test_forward_diff=false,
                    test_project_tangent=true,
                    test_injectivity_radius=false,
                    test_exp_log=false,
                    test_representation_size=true,
                )
            end

            TEST_STATIC_SIZED && @testset "Type $(MMatrix{6, 6, Float64, 36})" begin
                type = MMatrix{6,6,Float64,36}
                pts = convert.(type, points)
                test_manifold(
                    Sp_6,
                    pts;
                    retraction_methods=[CayleyRetraction(), ExponentialRetraction()],
                    default_retraction_method=CayleyRetraction(),
                    default_inverse_retraction_method=CayleyInverseRetraction(),
                    test_inplace=true,
                    is_point_atol_multiplier=1.0e7,
                    is_tangent_atol_multiplier=1.0e6,
                    retraction_atol_multiplier=1.0e4,
                    test_reverse_diff=false,
                    test_forward_diff=false,
                    test_project_tangent=false, # Cannot solve 'sylvester' for MMatrix-type.
                    test_injectivity_radius=false,
                    test_exp_log=false,
                    test_representation_size=true,
                )
            end
        end

        @testset "Gradient Computations" begin
            test_f(p) = tr(p)
            Q_grad = SymplecticMatrix(points[1])
            analytical_grad_f(p) = (1 / 2) * (p * Q_grad * p * Q_grad + p * p')

            p_grad = convert(Array{Float64}, points[1])
            ad_diff = RiemannianProjectionBackend(Manifolds.ForwardDiffBackend())

            @test isapprox(
                Manifolds.gradient(Sp_6, test_f, p_grad, ad_diff),
                analytical_grad_f(p_grad);
                atol=1.0e-16,
            )
            @test isapprox(
                Manifolds.gradient(Sp_6, test_f, p_grad, ad_diff; extended_metric=false),
                analytical_grad_f(p_grad);
                atol=1.0e-12,
            )

            grad_f_p = similar(p_grad)
            Manifolds.gradient!(Sp_6, test_f, grad_f_p, p_grad, ad_diff)
            @test isapprox(grad_f_p, analytical_grad_f(p_grad); atol=1.0e-16)

            Manifolds.gradient!(
                Sp_6,
                test_f,
                grad_f_p,
                p_grad,
                ad_diff;
                extended_metric=false,
            )
            @test isapprox(grad_f_p, analytical_grad_f(p_grad); atol=1.0e-12)
        end
    end

    @testset "SymplecticMatrix" begin
        # TODO: Test for different type matrices.
        @test SymplecticMatrix() == SymplecticMatrix(1)
        Sp_4 = Symplectic(4)
        pQ_1 = [
            0 0 -2 3
            0 0 1 -1
            -1 -1 0 0
            -3 -2 4 -4
        ]
        pQ_2 = [
            0 0 -2 5
            0 0 1 -2
            -2 -1 0 0
            -5 -2 4 -8
        ]
        p_odd_row = [
            0 0 -1 3
            0 0 1 0
            -2 -1 0 0
        ]
        p_odd_col = [
            0 -2 5
            0 1 -2
            -1 0 0
            -2 4 -8
        ]
        Q = SymplecticMatrix(pQ_1, pQ_2)
        Q2 = SymplecticMatrix(1)

        @testset "Type Basics" begin
            @test Q == Q2
            @test ndims(Q) == 2
            @test copy(Q) == Q
            @test eltype(SymplecticMatrix(1 // 1)) == Rational{Int64}
            @test convert(SymplecticMatrix{Float64}, Q) == SymplecticMatrix(1.0)
            @test "$Q" == "SymplecticMatrix{Int64}(): 1*[0 I; -I 0]"
            @test (
                "$(SymplecticMatrix(1 + 2im))" ==
                "SymplecticMatrix{Complex{Int64}}(): (1 + 2im)*[0 I; -I 0]"
            )
        end

        @testset "Matrix Operations" begin
            @test -Q == SymplecticMatrix(-1)
            @test (2 * Q) * (5 // 6) == SymplecticMatrix(5 // 3)

            @testset "Powers" begin
                @test inv(Q) * Q == I
                @test (
                    inv(SymplecticMatrix(-4.0 + 8im)) * SymplecticMatrix(-4.0 + 8im) ==
                    UniformScaling(1.0 + 0.0im)
                )
                @test Q * Q == -I
                @test Q^2 == -I
                @test Q^3 == -Q
                @test Q^4 == I
            end
            @testset "Addition (subtraction)" begin
                @test Q + Q == 2 * Q
                @test Q - SymplecticMatrix(1.0) == SymplecticMatrix(0.0)
                @test Q + pQ_1 == [
                    0 0 -1 3
                    0 0 1 0
                    -2 -1 0 0
                    -3 -3 4 -4
                ]
                @test Q - pQ_1 == [
                    0 0 3 -3
                    0 0 -1 2
                    0 1 0 0
                    3 1 -4 4
                ]
                @test pQ_1 - Q == [
                    0 0 -3 3
                    0 0 1 -2
                    0 -1 0 0
                    -3 -1 4 -4
                ]
                @test (pQ_1 + Q) == (Q + pQ_1)

                @test_throws ArgumentError Q + p_odd_row
            end
            @testset "Transpose-Adjoint" begin
                @test Q' == SymplecticMatrix(-1)
                @test transpose(SymplecticMatrix(10)) == SymplecticMatrix(-10)
                @test transpose(SymplecticMatrix(1 - 2.0im)) == SymplecticMatrix(-1 + 2.0im)
                @test adjoint(Q) == -Q
                @test adjoint(SymplecticMatrix(1 - 2.0im)) == SymplecticMatrix(-1 - 2.0im)
                @test adjoint(SymplecticMatrix(-1im)) == SymplecticMatrix(-1im)
                @test adjoint(SymplecticMatrix(2.0)) == SymplecticMatrix(-2.0)
            end
            @testset "Inplace mul!" begin
                z1 = [1 + 2im; 1 - 2im]
                @test lmul!(Q, copy(z1)) == Q * z1
                @test lmul!(Q, copy(p_odd_col)) == [
                    -1 0 0
                    -2 4 -8
                    0 2 -5
                    0 -1 2
                ]
                @test_throws ArgumentError lmul!(Q, copy(p_odd_row))

                @test rmul!(copy(z1'), Q) == z1' * Q
                @test rmul!(copy(p_odd_row), Q) == [
                    1 -3 0 0
                    -1 0 0 0
                    0 0 -2 -1
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
end
