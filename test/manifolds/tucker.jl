using ManifoldsBase: LinearAlgebra
include("../utils.jl")

@testset "Tucker" begin
    n⃗ = (4, 5, 6)
    r⃗ = (2, 3, 4)
    r⃗_small = (2, 3, 3)
    M = Tucker(n⃗, r⃗)
    M_small = Tucker(r⃗, (2, 3, 3))

    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)

    for T in types
        @testset "Type $T" begin
            U = ntuple(d -> Matrix{T}(LinearAlgebra.I, n⃗[d], r⃗[d]), 3)
            C₁ = reshape(
                T[
                    0.0 0.0 1.0 0.0
                    1.0 1/√2 0.0 0.0
                    1.0 -1/√2 0.0 0.0
                    0.0 0.0 0.0 0.0
                    0.0 0.0 0.0 1.0
                    0.0 0.0 0.0 0.0
                ],
                (2, 3, 4),
            )
            C₂ = reshape(
                T[
                    0.0 0.0 -1/√2 1/√2
                    -1/√2 -0.5 0.0 0.0
                    1.0 -1/√2 0.0 0.0
                    0.0 0.0 0.0 0.0
                    0.0 0.0 1/√2 1/√2
                    1/√2 0.5 0.0 0.0
                ],
                (2, 3, 4),
            )
            C₃ = reshape(
                T[
                    1/√2 0.5 1/√2 0.0
                    1/√2 0.5 -1/√2 0.0
                    1/√2 -0.5 0.0 0.0
                    -1/√2 0.5 0.0 0.0
                    0.0 0.0 0.0 1/√2
                    0.0 0.0 0.0 -1/√2
                ],
                (2, 3, 4),
            )

            p = TuckerPoint(C₁, U...)
            p_small = TuckerPoint(C₂, r⃗_small)
            p_ambient = embed(M, p)

            U⊥ = ntuple(d -> nullspace(U[d]') * Matrix{Bool}(I, n⃗[d] - r⃗[d], r⃗[d]), 3)
            v = TuckerTVector(reshape(collect(1.0:prod(r⃗)), r⃗), U⊥)
            U⊥ = ntuple(d -> nullspace(U[d]') * Matrix{Bool}(I, n⃗[d] - r⃗[d], r⃗[d]), 3)
            w = TuckerTVector(reshape((1.0:prod(r⃗)) .^ 2, r⃗), U⊥)

            norm_v = norm(M, p, v)
            @test norm(M, p, 2 * v - v * 2) ≤ √eps(T) * norm_v
            @test norm(M, p, (v / 2) - (2 \ v)) ≤ eps(T) * norm_v
            @test norm(M, p, (v / 2) - 0.5v) ≤ eps(T) * norm_v
            @test +v == v

            @test embed(M, p) ≈ reshape(kron(reverse(U)...) * vec(C₁), n⃗)
            @test embed(M, TuckerPoint(p_ambient, r⃗)) ≈ p_ambient
            @test inner(M, p, v, w) ≈ dot(embed(M, p, v), embed(M, p, w))
            @test inner(M, p, v, v) ≈ norm(M, p, v)^2
            w_ambient = embed(M, p, w)
            @test inner(M, p, v, w_ambient) ≈ inner(M, p, v, w) ≈ inner(M, p, w_ambient, v)

            @test p ≈ TuckerPoint(p_ambient, r⃗)
            @test v == TuckerTVector(v.Ċ, v.U̇)

            @test manifold_dimension(M) ==
                  prod(r⃗) + sum(ntuple(d -> r⃗[d] * (n⃗[d] - r⃗[d]), length(r⃗)))

            @test is_point(M_small, p_small)
            @test is_point(M, p)
            @test is_point(M, embed(M, p))
            @test !is_point(M, zeros(n⃗))
            @test !is_point(M, TuckerPoint(zeros(T, r⃗), U...))
            @test !is_point(M, TuckerPoint(C₁, map(rk -> Matrix{T}(I, rk, rk), r⃗)...))
            @test !is_point(M, p_small)
            @test !is_point(M, embed(M_small, p_small))
            @test !is_point(M_small, p)
            q = allocate(p)
            copyto!(q, p)
            q.hosvd.core .= 2 * p.hosvd.core
            @test !is_point(M, q)

            u = allocate(v)
            u .= v
            @test u == v
            u = allocate(v)
            copyto!(u, v)
            @test u == v

            # broadcasting
            @test axes(v) === ()
            u = copy(v)
            # test that the copy is equal to the original, but represented by
            # a new array
            @test u.U̇ !== v.U̇
            @test u.Ċ == v.Ċ
            x = u .+ v .* 2
            @test x isa TuckerTVector
            @test x == v + u * 2
            x .= 2 .* u .+ v
            @test x == 2 * u + v

            @test is_vector(M, p, v)
            @test !is_vector(M, p_small, v)
            @test !is_vector(
                M,
                p,
                TuckerTVector(ones(T, r⃗), map(u⊥ -> u⊥[:, 1:(end - 1)], U⊥)),
            )

            ℬ = get_basis(M, p, DefaultOrthonormalBasis())
            J = convert(Matrix, ℬ)
            @test J'J ≈ LinearAlgebra.I
            vectors = get_vectors(M, p, DefaultOrthonormalBasis())
            vectors_mtx = hcat(map(ξ -> vec(embed(M, p, ξ)), vectors)...)
            @test J ≈ vectors_mtx

            shiftprint(x) = replace(sprint(show, "text/plain", x), "\n" => "\n ")

            @test sprint(show, "text/plain", M) == "Tucker((4, 5, 6), (2, 3, 4), ℝ)"
            @test sprint(show, "text/plain", p) == """
            $(summary(p))
            U factor 1:
             $(shiftprint(p.hosvd.U[1]))\n
            U factor 2:
             $(shiftprint(p.hosvd.U[2]))\n
            U factor 3:
             $(shiftprint(p.hosvd.U[3]))\n
            Core:
             $(shiftprint(p.hosvd.core))"""

            @test sprint(show, "text/plain", v) == """
            $(summary(v))
            U̇ factor 1:
             $(shiftprint(v.U̇[1]))\n
            U̇ factor 2:
             $(shiftprint(v.U̇[2]))\n
            U̇ factor 3:
             $(shiftprint(v.U̇[3]))\n
            Ċ factor:
             $(shiftprint(v.Ċ))"""

            @test sprint(show, "text/plain", ℬ) == """
            $(summary(ℬ)) ≅ $(shiftprint(J))
            """

            pts = [
                p,
                TuckerPoint(C₂, U[1][[4, 2, 1, 3], :], U[2:end]...),
                TuckerPoint(C₃, U[1][[4, 3, 1, 2], :], U[2:end]...),
            ]
            test_manifold(
                M,
                pts;
                is_mutating=false, # avoid allocations of the wrong type
                basis_types_to_from=(DefaultOrthonormalBasis(),),
                basis_types_vecs=(DefaultOrthonormalBasis(),),
                test_exp_log=false,
                default_inverse_retraction_method=ProjectionInverseRetraction(),
                test_injectivity_radius=false,
                default_retraction_method=PolarRetraction(),
                test_is_tangent=false,
                test_project_tangent=false,
                test_default_vector_transport=false,
                test_vector_spaces=false,
                test_vee_hat=false,
                test_tangent_vector_broadcasting=true,
                test_representation_size=false,
                projection_atol_multiplier=15,
                retraction_methods=[PolarRetraction()],
                inverse_retraction_methods=[ProjectionInverseRetraction()],
                mid_point12=nothing,
            )
        end
    end
end
