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

        @testset "Euclidean — basic GPU operations" begin
            M = Euclidean(4)
            p = CuArray(randn(4))
            X = CuArray(randn(4))
            Y = CuArray(randn(4))

            # exp / retract
            q = exp(M, p, X)
            @test q isa CuArray
            @test isapprox(Array(q), Array(p) + Array(X))

            # inner / norm
            @test inner(M, p, X, Y) isa Real
            @test norm(M, p, X) >= 0

            # project
            Xp = project(M, p, X)
            @test Xp isa CuArray
            @test isapprox(Array(Xp), Array(X))
        end

        @testset "Euclidean matrix — GPU operations" begin
            M = Euclidean(3, 3)
            P = CuArray(randn(3, 3))
            X = CuArray(randn(3, 3))

            Q = exp(M, P, X)
            @test Q isa CuArray{Float64,2}
            @test size(Q) == (3, 3)
            @test isapprox(Array(Q), Array(P) + Array(X))
        end

        @testset "Sphere — GPU operations" begin
            M = Sphere(3)
            # Create a valid point on S^3
            p_raw = randn(4)
            p_raw ./= norm(p_raw)
            p = CuArray(p_raw)

            # Random tangent vector
            X_raw = randn(4)
            X_raw .-= dot(X_raw, p_raw) * p_raw  # project onto tangent space
            X = CuArray(X_raw)

            # exp
            q = exp(M, p, X)
            @test q isa CuArray
            # Result should be on the sphere
            @test isapprox(norm(Array(q)), 1.0; atol=1e-10)

            # inner / norm
            @test inner(M, p, X, X) isa Real
            @test norm(M, p, X) >= 0

            # project point
            p_proj = project(M, CuArray(randn(4)))
            @test p_proj isa CuArray
            @test isapprox(norm(Array(p_proj)), 1.0; atol=1e-10)

            # project vector
            V = CuArray(randn(4))
            Xp = project(M, p, V)
            @test Xp isa CuArray
            # Should be orthogonal to p
            @test abs(dot(Array(Xp), Array(p))) < 1e-10
        end

        @testset "UnitaryMatrices(2) — GPU exp and project" begin
            M = UnitaryMatrices(2)

            # Create a valid unitary matrix on GPU
            A = randn(ComplexF64, 2, 2)
            U, _, V = svd(A)
            p_cpu = U * V'
            p = CuArray(p_cpu)

            # Create a skew-Hermitian tangent vector
            B = randn(ComplexF64, 2, 2)
            X_cpu = (B - B') / 2  # skew-Hermitian
            X = CuArray(X_cpu)

            # exp: p * exp(X) — uses matrix exponential
            q = exp(M, p, X)
            @test q isa CuArray{ComplexF64,2}
            @test size(q) == (2, 2)
            # Result should be unitary: q'q ≈ I
            q_cpu = Array(q)
            @test isapprox(q_cpu' * q_cpu, I(2); atol=1e-10)

            # project tangent vector
            V = CuArray(randn(ComplexF64, 2, 2))
            Xp = project(M, p, V)
            @test Xp isa CuArray{ComplexF64,2}
            # Result should be skew-Hermitian: X + X' ≈ 0
            Xp_cpu = Array(Xp)
            @test isapprox(Xp_cpu + Xp_cpu', zeros(2, 2); atol=1e-10)
        end

        @testset "UnitaryMatrices(2) — polar retraction on GPU" begin
            M = UnitaryMatrices(2)

            # Valid unitary point
            A = randn(ComplexF64, 2, 2)
            U, _, V = svd(A)
            p_cpu = U * V'
            p = CuArray(p_cpu)

            # Small skew-Hermitian tangent vector
            B = 0.1 * randn(ComplexF64, 2, 2)
            X_cpu = (B - B') / 2
            X = CuArray(X_cpu)

            # Polar retraction (default for Stiefel/Unitary)
            q = retract(M, p, X, PolarRetraction())
            @test q isa CuArray{ComplexF64,2}
            q_cpu = Array(q)
            @test isapprox(q_cpu' * q_cpu, I(2); atol=1e-10)
        end

        @testset "UnitaryMatrices(2) — QR retraction on GPU" begin
            M = UnitaryMatrices(2)

            A = randn(ComplexF64, 2, 2)
            U, _, V = svd(A)
            p_cpu = U * V'
            p = CuArray(p_cpu)

            B = 0.1 * randn(ComplexF64, 2, 2)
            X_cpu = (B - B') / 2
            X = CuArray(X_cpu)

            q = retract(M, p, X, QRRetraction())
            @test q isa CuArray{ComplexF64,2}
            q_cpu = Array(q)
            # Result should be unitary
            @test isapprox(q_cpu' * q_cpu, I(2); atol=1e-10)
        end

        @testset "UnitaryMatrices — log on GPU (complex)" begin
            M = UnitaryMatrices(2)

            # Two unitary matrices
            A1 = randn(ComplexF64, 2, 2)
            U1, _, V1 = svd(A1)
            p = CuArray(U1 * V1')

            A2 = randn(ComplexF64, 2, 2)
            U2, _, V2 = svd(A2)
            q = CuArray(U2 * V2')

            X = log(M, p, q)
            @test X isa CuArray{ComplexF64,2}
            # Should be skew-Hermitian
            X_cpu = Array(X)
            @test isapprox(X_cpu + X_cpu', zeros(2, 2); atol=1e-8)
        end

        @testset "Float32 operations on GPU" begin
            M = Euclidean(8)
            p = CuArray(randn(Float32, 8))
            X = CuArray(randn(Float32, 8))

            q = exp(M, p, X)
            @test q isa CuArray{Float32,1}
            @test isapprox(Array(q), Array(p) + Array(X); atol=1e-5)

            n = norm(M, p, X)
            @test n isa Real
        end

        @testset "PowerManifold on GPU" begin
            M_base = Euclidean(3)
            M = PowerManifold(M_base, 4)

            # PowerManifold with NestedPowerRepresentation uses arrays of arrays
            p = [CuArray(randn(3)) for _ in 1:4]
            X = [CuArray(randn(3)) for _ in 1:4]

            q = exp(M, p, X)
            @test length(q) == 4
            for i in 1:4
                @test q[i] isa CuArray{Float64,1}
                @test isapprox(Array(q[i]), Array(p[i]) + Array(X[i]))
            end
        end
    else
        @info "CUDA not functional, skipping ManifoldsCUDAExt tests"
    end
end
