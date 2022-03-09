include("../utils.jl")

function Ω(::SymplecticStiefel, p, X)
    Q = SymplecticMatrix(X, p)
    pT_p = lu(p' * p)

    inv_pTp_pT = pT_p \ p'
    X_inv_pTp_pT = X * inv_pTp_pT

    Ω = Q * (X_inv_pTp_pT') * (I - Q' * p * inv_pTp_pT * Q) * Q
    Ω .+= X_inv_pTp_pT
    return Ω
end

function exp_naiive!(M::SymplecticStiefel, q, p, X)
    Ω_X = Ω(M, p, X)
    q .= exp(Ω_X - Ω_X') * exp(Array(Ω_X')) * p
    return q
end

@testset "SymplecticStiefel" begin
    @testset "Real" begin
        SpSt_6_4 = SymplecticStiefel(2 * 3, 2 * 2)

        p_6_4 = Array{Float64}(
            [
                0 0 -5 -1
                0 0 9 -2
                0 0 -2 1
                -2 -9 -3 6
                -3 -13 -21 9
                -8 -36 18 -6
            ],
        )
        q_6_4 = Array{Float64}([
            0 0 -3 1
            0 0 8 -3
            0 0 -2 1
            -1 -4 -6 3
            -1 -3 -21 9
            -2 -6 18 -6
        ])
        X1 = [
            0.0 0.0 4.25 4.25
            0.0 0.0 0.125 0.125
            0.0 0.0 -1.125 -1.125
            3.625 18.125 -10.875 -10.875
            5.0 25.0 -9.0 -9.0
            13.5 67.5 4.5 4.5
        ]
        X2 = [
            -0.02648060 0.00416977 0.01130802 0.01015956
            0.01718954 -0.00680433 0.02364406 -0.00083272
            0.00050392 0.00191916 -0.01035902 -0.00079734
            0.01811917 -0.02307032 -0.04297277 -0.05409099
            -0.02529516 0.00959934 -0.08594555 -0.06117803
            -0.02823014 0.00029946 -0.04196034 -0.04145413
        ]

        points = [
            [
                0 0 2 6
                0 0 1 -4
                0 0 0 1
                -1 0 -7 1
                1 0 -4 -24
                10 -1 4 -4
            ],
            [
                0 0 5 8
                0 0 -2 -6
                0 0 0 1
                1 0 -4 3
                3 0 -19 -34
                10 -1 1 -6
            ],
            [
                5 -3 0 0
                11 -6 0 0
                -2 1 0 0
                8 -5 -3 -3
                0 0 2 1
                5 -3 3 -2
            ],
            [
                1 -1 0 0
                3 -2 0 0
                0 0 0 0
                2 -2 -2 -3
                0 0 1 1
                1 -1 1 -2
            ],
            [
                0 0 0 0
                0 0 -1 0
                0 0 4 -1
                -1 -10 -5 2
                1 4 -2 0
                0 1 -16 4
            ],
            [
                0 0 0 0
                0 0 -1 0
                0 0 4 -1
                0 -6 -5 2
                1 4 -2 0
                0 1 -16 4
            ],
        ]

        close_points = [
            [
                -2 1 -2 -14
                1 0 1 11
                1 0 4 9
                0 0 0 1
                0 0 0 -1
                0 0 1 3
            ],
            [
                -3 1 -2 -10
                1 0 1 7
                1 0 4 10
                0 0 0 1
                0 0 0 0
                0 0 1 3
            ],
            [
                -2 1 -6 -26
                1 0 5 23
                3 -1 2 3
                0 0 1 4
                0 0 0 -1
                0 0 1 3
            ],
        ]

        @testset "Basics" begin
            @test repr(SpSt_6_4) == "SymplecticStiefel{6, 4, ℝ}()"
            @test representation_size(SpSt_6_4) == (6, 4)
            @test base_manifold(SpSt_6_4) === SpSt_6_4

            @test is_point(SpSt_6_4, p_6_4)
            @test_throws DomainError is_point(SpSt_6_4, 2 * p_6_4, true)

            @test is_vector(SpSt_6_4, p_6_4, X1; atol=1.0e-12)
            @test is_vector(SpSt_6_4, p_6_4, X2; atol=1.0e-6)
            @test_throws DomainError is_vector(SpSt_6_4, p_6_4, X2, true; atol=1.0e-12)
            @test is_vector(SpSt_6_4, p_6_4, X1 + X2; atol=1.0e-6)
            @test_throws DomainError is_vector(SpSt_6_4, p_6_4, X1 + p_6_4, true)
        end
        @testset "Symplectic Inverse" begin
            I_2k = Array(I, 4, 4)
            @test Manifolds.symplectic_inverse_times(SpSt_6_4, p_6_4, p_6_4) == I_2k
            @test Manifolds.symplectic_inverse_times!(
                SpSt_6_4,
                zeros(4, 4),
                p_6_4,
                p_6_4,
            ) == I_2k
            @test inv(SpSt_6_4, p_6_4) * p_6_4 == I_2k
            @test inv!(SpSt_6_4, copy(p_6_4'), p_6_4) * p_6_4 == I_2k
        end
        @testset "Embedding" begin
            x = [
                1 1 -9 7
                -1 -1 -13 11
                7 8 -22 19
                0 0 7 -6
                0 0 -1 1
                0 0 -1 1
            ]
            y = similar(x)
            z = embed(SpSt_6_4, x)
            @test z == x

            Y = similar(X1)
            embed!(SpSt_6_4, Y, p_6_4, X1)
            @test Y == X1
        end
        @testset "Retractions and Exponential Mapping" begin
            @test isapprox(retract(SpSt_6_4, p_6_4, X1), q_6_4; atol=1.0e-12)
            @test isapprox(
                retract(SpSt_6_4, p_6_4, X1, CayleyRetraction()),
                q_6_4;
                atol=1.0e-12,
            )

            X_inv_cayley_retraction = inverse_retract(SpSt_6_4, p_6_4, q_6_4)
            X_inv_cayley_retraction_2 =
                inverse_retract(SpSt_6_4, p_6_4, q_6_4, CayleyInverseRetraction())
            @test isapprox(X_inv_cayley_retraction, X_inv_cayley_retraction_2; atol=1.0e-16)
            @test isapprox(X_inv_cayley_retraction, X1; atol=1.0e-12)
        end
        @testset "Riemannian Metric" begin
            X1_norm = 37.85466645
            @test isapprox(norm(SpSt_6_4, p_6_4, X1), X1_norm; atol=1.0e-8)
            @test isapprox(
                norm(SpSt_6_4, p_6_4, X1),
                √inner(SpSt_6_4, p_6_4, X1, X1);
                atol=1.0e-8,
            )

            X2_norm = 1.0
            @test isapprox(norm(SpSt_6_4, p_6_4, X2), X2_norm; atol=1.0e-6)
            @test isapprox(
                norm(SpSt_6_4, p_6_4, X2),
                √inner(SpSt_6_4, p_6_4, X2, X2);
                atol=1.0e-6,
            )

            # Project Project matrix A ∈ ℝ^{6 × 4} onto (T_pSpSt):
            A_6_4 = Array{Float64}([
                -7 2 12 0
                4 0 1 -2
                -1 -1 4 0
                -18 4 -1 5
                7 0 -2 11
                2 2 -2 9
            ])
            A_6_4_proj = similar(A_6_4)
            Manifolds.project!(SpSt_6_4, A_6_4_proj, p_6_4, A_6_4)
            @test is_vector(SpSt_6_4, p_6_4, A_6_4_proj, true; atol=1.0e-12)
        end
        @testset "Generate random points/tangent vectors" begin
            M_big = SymplecticStiefel(20, 10)
            p_big = rand(M_big)
            @test is_point(M_big, p_big, true; atol=1.0e-14)
            X_big = rand(M_big; vector_at=p_big, hamiltonian_norm=1.0)
            @test is_vector(M_big, p_big, X_big, true; atol=1.0e-14)
        end
        @testset "test_manifold(Symplectic(6), ...)" begin
            types = [Matrix{Float64}]
            TEST_FLOAT32 && push!(types, Matrix{Float32})
            TEST_STATIC_SIZED && push!(types, MMatrix{6,4,Float64,24})
            for type in types
                @testset "Type $(type)" begin
                    @testset "CayleyRetraction" begin
                        test_manifold(
                            SpSt_6_4,
                            convert.(type, points);
                            retraction_methods=[CayleyRetraction()],
                            default_retraction_method=CayleyRetraction(),
                            default_inverse_retraction_method=CayleyInverseRetraction(),
                            test_inplace=true,
                            is_point_atol_multiplier=1.0e4,
                            is_tangent_atol_multiplier=1.0e3,
                            retraction_atol_multiplier=1.0e1,
                            test_reverse_diff=false,
                            test_forward_diff=false,
                            test_project_tangent=(type != MMatrix{6,4,Float64,24}),
                            test_injectivity_radius=false,
                            test_exp_log=false,
                            test_representation_size=true,
                        )
                    end

                    @testset "ExponentialRetraction" begin
                        test_manifold(
                            SpSt_6_4,
                            convert.(type, close_points);
                            retraction_methods=[ExponentialRetraction()],
                            default_retraction_method=ExponentialRetraction(),
                            default_inverse_retraction_method=CayleyInverseRetraction(),
                            test_inplace=true,
                            is_point_atol_multiplier=1.0e11,
                            is_tangent_atol_multiplier=1.0e2,
                            retraction_atol_multiplier=1.0e4,
                            test_reverse_diff=false,
                            test_forward_diff=false,
                            test_project_tangent=(type != MMatrix{6,4,Float64,24}),
                            test_injectivity_radius=false,
                            test_exp_log=false,
                            test_representation_size=true,
                        )
                    end
                end
            end # for
        end

        @testset "Gradient Computations" begin
            Q_grad = SymplecticMatrix(points[1])
            function test_f(p)
                k = size(p)[2]
                return tr(p[1:k, 1:k])
            end
            function analytical_grad_f(p)
                n, k = size(p)
                euc_grad_f = [Array(I, k, k); zeros((n - k), k)]
                return Q_grad * p * (euc_grad_f') * Q_grad * p + euc_grad_f * p' * p
            end
            p_grad = convert(Array{Float64}, points[1])
            ad_diff = RiemannianProjectionBackend(Manifolds.ForwardDiffBackend())

            @test isapprox(
                Manifolds.gradient(SpSt_6_4, test_f, p_grad, ad_diff),
                analytical_grad_f(p_grad);
                atol=1.0e-16,
            )

            grad_f_p = similar(p_grad)
            Manifolds.gradient!(SpSt_6_4, test_f, grad_f_p, p_grad, ad_diff)
            @test isapprox(grad_f_p, analytical_grad_f(p_grad); atol=1.0e-16)
        end
    end
end
