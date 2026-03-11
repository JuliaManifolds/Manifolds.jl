using Manifolds, ManifoldsBase, Test
using LinearAlgebra

@testset "ManifoldsCUDAExt" begin
    cuda_loaded = false
    try
        using CUDA
        cuda_loaded = CUDA.functional()
    catch
        cuda_loaded = false
    end

    if cuda_loaded
        @eval using CUDA

        @testset "Euclidean — CPU vs GPU" begin
            M = Euclidean(4)
            p_cpu = randn(4)
            X_cpu = randn(4)
            Y_cpu = randn(4)
            p = CuArray(p_cpu)
            X = CuArray(X_cpu)
            Y = CuArray(Y_cpu)

            # exp
            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test q isa CuArray
            @test isapprox(Array(q), q_cpu; atol=1e-14)

            # project point (identity on Euclidean)
            pp = project(M, p)
            @test pp isa CuArray
            @test isapprox(Array(pp), p_cpu; atol=1e-14)

            # project vector
            Xp = project(M, p, X)
            @test Xp isa CuArray
            @test isapprox(Array(Xp), X_cpu; atol=1e-14)

            # inner
            ip = inner(M, p, X, Y)
            ip_cpu = inner(M, p_cpu, X_cpu, Y_cpu)
            @test isapprox(ip, ip_cpu; atol=1e-14)

            # norm
            n = norm(M, p, X)
            n_cpu = norm(M, p_cpu, X_cpu)
            @test isapprox(n, n_cpu; atol=1e-14)

            # Note: distance() uses @simd scalar indexing, not GPU-compatible
        end

        @testset "Euclidean matrix — CPU vs GPU" begin
            M = Euclidean(3, 3)
            P_cpu = randn(3, 3)
            X_cpu = randn(3, 3)
            P = CuArray(P_cpu)
            X = CuArray(X_cpu)

            Q = exp(M, P, X)
            Q_cpu = exp(M, P_cpu, X_cpu)
            @test Q isa CuArray{Float64,2}
            @test isapprox(Array(Q), Q_cpu; atol=1e-14)
        end

        @testset "Euclidean Float32 — CPU vs GPU" begin
            M = Euclidean(8)
            p_cpu = randn(Float32, 8)
            X_cpu = randn(Float32, 8)
            p = CuArray(p_cpu)
            X = CuArray(X_cpu)

            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test q isa CuArray{Float32,1}
            @test isapprox(Array(q), q_cpu; atol=1e-5)

            n = norm(M, p, X)
            n_cpu = norm(M, p_cpu, X_cpu)
            @test isapprox(n, n_cpu; atol=1e-5)
        end

        @testset "Sphere — CPU vs GPU" begin
            M = Sphere(3)
            p_cpu = randn(4); p_cpu ./= norm(p_cpu)
            X_cpu = randn(4); X_cpu .-= dot(X_cpu, p_cpu) * p_cpu
            p = CuArray(p_cpu)
            X = CuArray(X_cpu)

            # exp
            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test q isa CuArray
            @test isapprox(norm(Array(q)), 1.0; atol=1e-10)
            @test isapprox(Array(q), q_cpu; atol=1e-10)

            # inner / norm
            ip = inner(M, p, X, X)
            ip_cpu = inner(M, p_cpu, X_cpu, X_cpu)
            @test isapprox(ip, ip_cpu; atol=1e-10)

            # project point
            p_proj = project(M, CuArray(randn(4)))
            @test p_proj isa CuArray
            @test isapprox(norm(Array(p_proj)), 1.0; atol=1e-10)

            # project vector
            V = CuArray(randn(4))
            Xp = project(M, p, V)
            @test Xp isa CuArray
            @test abs(dot(Array(Xp), p_cpu)) < 1e-10

            # distance
            q2_cpu = randn(4); q2_cpu ./= norm(q2_cpu)
            d = distance(M, p, CuArray(q2_cpu))
            d_cpu = distance(M, p_cpu, q2_cpu)
            @test isapprox(d, d_cpu; atol=1e-10)
        end

        @testset "UnitaryMatrices(2) — CPU vs GPU" begin
            M = UnitaryMatrices(2)

            # Create a valid unitary matrix
            A = randn(ComplexF64, 2, 2)
            U, _, V = svd(A)
            p_cpu = U * V'
            p = CuArray(p_cpu)

            # Skew-Hermitian tangent vector
            B = 0.1 * randn(ComplexF64, 2, 2)
            X_cpu = (B - B') / 2
            X = CuArray(X_cpu)

            # exp
            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test q isa CuArray{ComplexF64,2}
            q_arr = Array(q)
            @test isapprox(q_arr' * q_arr, I(2); atol=1e-10)
            @test isapprox(q_arr, q_cpu; atol=1e-10)

            # project tangent vector
            V_rand = CuArray(randn(ComplexF64, 2, 2))
            Xp = project(M, p, V_rand)
            @test Xp isa CuArray{ComplexF64,2}
            Xp_arr = Array(Xp)
            @test isapprox(Xp_arr + Xp_arr', zeros(2, 2); atol=1e-10)

            # Note: PolarRetraction uses project!(M, q, p; check_det=true) internally
            # which doesn't match the GPU dispatch. Skipping until upstream fix.

            # QR retraction
            q_qr = retract(M, p, X, QRRetraction())
            @test q_qr isa CuArray{ComplexF64,2}
            q_qr_arr = Array(q_qr)
            @test isapprox(q_qr_arr' * q_qr_arr, I(2); atol=1e-10)
        end

        @testset "UnitaryMatrices(2) — log on GPU (complex)" begin
            M = UnitaryMatrices(2)

            A1 = randn(ComplexF64, 2, 2)
            U1, _, V1 = svd(A1)
            p = CuArray(U1 * V1')

            A2 = randn(ComplexF64, 2, 2)
            U2, _, V2 = svd(A2)
            q = CuArray(U2 * V2')

            X = log(M, p, q)
            @test X isa CuArray{ComplexF64,2}
            X_arr = Array(X)
            @test isapprox(X_arr + X_arr', zeros(2, 2); atol=1e-8)
        end

        @testset "Grassmann — CPU vs GPU" begin
            M = Grassmann(4, 2)

            # Create a valid point on Gr(4,2): orthonormal 4x2 matrix
            A_cpu = randn(4, 2)
            p_cpu = Matrix(qr(A_cpu).Q)[:, 1:2]
            p = CuArray(p_cpu)

            # Tangent vector: must satisfy p'X = 0
            Z = randn(4, 2)
            X_cpu = Z - p_cpu * (p_cpu' * Z)
            X = CuArray(X_cpu)

            # exp
            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test q isa CuArray{Float64,2}
            q_arr = Array(q)
            @test isapprox(q_arr' * q_arr, I(2); atol=1e-10)
            @test isapprox(q_arr, q_cpu; atol=1e-10)

            # project
            V = CuArray(randn(4, 2))
            Xp = project(M, p, V)
            @test Xp isa CuArray{Float64,2}
            @test isapprox(Array(p)' * Array(Xp), zeros(2, 2); atol=1e-10)

            # retract (QR) — use small tangent vector
            small_X = CuArray(0.1 .* X_cpu)
            q_qr = retract(M, p, small_X, QRRetraction())
            @test q_qr isa CuArray{Float64,2}
            q_qr_arr = Array(q_qr)
            @test isapprox(q_qr_arr' * q_qr_arr, I(2); atol=1e-10)
        end

        # Note: Hyperbolic manifold uses minkowski_metric with scalar indexing (a[1]*b[1]),
        # which is not GPU-compatible. Skipping until a GPU-friendly implementation exists.

        @testset "PowerManifold on GPU (nested)" begin
            M_base = Euclidean(3)
            M = PowerManifold(M_base, NestedPowerRepresentation(), 4)

            p = [CuArray(randn(3)) for _ in 1:4]
            X = [CuArray(randn(3)) for _ in 1:4]
            p_cpu = [Array(pi) for pi in p]
            X_cpu = [Array(xi) for xi in X]

            q = exp(M, p, X)
            q_cpu = exp(M, p_cpu, X_cpu)
            @test length(q) == 4
            for i in 1:4
                @test q[i] isa CuArray{Float64,1}
                @test isapprox(Array(q[i]), q_cpu[i]; atol=1e-14)
            end
        end
    else
        @info "CUDA not functional, skipping ManifoldsCUDAExt tests"
    end
end
