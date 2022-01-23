include("../utils.jl")

@testset "Symplectic" begin
    @testset "Real" begin
        small_n = 1
        M = Symplectic(2 * small_n)
        M2 = MetricManifold(M, EuclideanMetric())
        p = [0.0 1.0/2.0; -2.0 -2.0]
        X1 = [
            -0.121212 0.121212
            0.969697 -1.0
        ]
        X2 = [
            0.0 0.0
            0.0 -1.0
        ]

        @testset "Basics" begin
            @test repr(M) == "Symplectic($(2*small_n), ℝ)"
            @test representation_size(M) == (2small_n, 2small_n)
            @test base_manifold(M) === M

            @test is_point(M, p)
            @test !is_point(M, p + I)

            @test is_vector(M, p, X1; atol=1.0e-6)
            @test is_vector(M, p, X2; atol=1.0e-12)
            @test is_vector(M, p, X1 + X2; atol=1.0e-6)
            @test !is_vector(M, p, X1 + [0.1 0.1; -0.1 0.1]; atol=1.0e-6)
        end
        @testset "Symplectic Inverse" begin
            I_2n = Array(I, 2 * small_n, 2 * small_n)
            @test Manifolds.symplectic_inverse_times(M, p, p) == I_2n
            @test inv!(M, copy(p)) * p == I_2n
        end
        @testset "Embedding and Projection" begin
            x = [0.0 1.0/2.0; -2.0 -2.0]
            y = similar(x)
            z = embed(M, x)
            @test z == x

            Y = similar(X1)
            embed!(M, Y, p, X1)
            @test Y == X1
        end
        @testset "Retractions and Exponential Mapping" begin
            q_exp = [
                -0.0203171 0.558648
                -1.6739 -3.19344
            ]
            @test isapprox(exp(M, p, X2), q_exp; atol=1.0e-5)
            @test isapprox(retract(M, p, X2, ExponentialRetraction()), q_exp; atol=1.0e-5)

            q_cay = [
                0.0 0.5
                -2.0 -3.0
            ]
            @test retract(M, p, X2) == q_cay
            @test retract(M, p, X2, CayleyRetraction()) == q_cay

            X_inv_cayley_retraction = inverse_retract(M, p, q_cay)
            X_inv_cayley_retraction_2 =
                inverse_retract(M, p, q_cay, CayleyInverseRetraction())
            @test X_inv_cayley_retraction == X_inv_cayley_retraction_2
            @test X_inv_cayley_retraction ≈ X2
        end

        @testset "Riemannian metric" begin
            X1_p_norm = 0.49259905148939337
            @test norm(M, p, X1) == X1_p_norm
            @test norm(M, p, X1) == √(inner(M, p, X1, X1))

            X2_p_norm = 1 / 2
            @test norm(M, p, X2) == X2_p_norm
            @test norm(M, p, X2) == √(inner(M, p, X2, X2))
        end

        @testset "Generate random points/tangent vectors" begin
            M_big = Symplectic(20)
            p_big = rand(M_big)
            @test is_point(M_big, p_big, true; atol=1.0e-12)
            X_big = rand(M_big, p_big)
            @test is_vector(M_big, p_big, X_big, true; atol=1.0e-12)
        end

        @testset "test_manifold(Symplectic(6), ...)" begin
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
            points = map(x -> Array{Float64}(x), points)
            test_manifold(
                Sp_6,
                points;
                retraction_methods=[CayleyRetraction(), ExponentialRetraction()],
                default_inverse_retraction_method=CayleyInverseRetraction(),
                test_inplace=true,
                is_point_atol_multiplier=1.0e6,
                is_tangent_atol_multiplier=1.0e4,
                retraction_atol_multiplier=1.0e4,
                test_reverse_diff=false,
                test_forward_diff=false,
                test_project_tangent=true,
                test_injectivity_radius=false,
                test_exp_log=false,
                test_representation_size=true,
            )
        end
    end

    @testset "SymplecticMatrix" begin
        # TODO: Test for different type matrices.
        M = Symplectic(4)
        p1 = [
            0 0 -2 3
            0 0 1 -1
            -1 -1 0 0
            -3 -2 4 -4
        ]
        p2 = [
            0 0 -2 5
            0 0 1 -2
            -2 -1 0 0
            -5 -2 4 -8
        ]
        Q = SymplecticMatrix(p1, p2)
        Q2 = SymplecticMatrix(1)

        @testset "Type Basics" begin
            @test Q == Q2
            @test ndims(Q) == 2
            @test copy(Q) == Q
            @test eltype(SymplecticMatrix(1 // 1)) == Rational{Int64}
            @test convert(SymplecticMatrix{Float64}, Q) == SymplecticMatrix(1.0)
            @test "$Q" == "SymplecticMatrix{Int64}(): 1*[0 I; -I 0]"
            @test (
                "$(SymplecticMatrix(1.0 + 2.0im))" ==
                "SymplecticMatrix{ComplexF64}(): (1.0 + 2.0im)*[0 I; -I 0]"
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
                @test Q + p1 == [
                    0 0 -1 3
                    0 0 1 0
                    -2 -1 0 0
                    -3 -3 4 -4
                ]
                @test Q - p1 == [
                    0 0 3 -3
                    0 0 -1 2
                    0 1 0 0
                    3 1 -4 4
                ]
                @test p1 - Q == [
                    0 0 -3 3
                    0 0 1 -2
                    0 -1 0 0
                    -3 -1 4 -4
                ]
                @test (p1 + Q) == (Q + p1)
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
                @test rmul!(copy(z1'), Q) == z1' * Q

                z1_copy = copy(z1)
                mul!(z1_copy, Q, z1)
                @test z1_copy == Q * z1

                mul!(z1_copy', z1', Q)
                @test z1_copy' == (z1' * Q)
            end
        end

        @testset "Symplectic Inverse Ops." begin
            @test ((Q' * p1' * Q) * p1 - I) == zeros(eltype(p1), size(p1)...)
        end
    end
end
