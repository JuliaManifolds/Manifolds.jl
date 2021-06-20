include("utils.jl")

@testset "Tucker" begin
    Random.seed!(1)

    n⃗ = (4, 5, 6)
    r⃗ = (2, 3, 4)
    M = Tucker(n⃗, r⃗)

    types = [Float64]
    TEST_FLOAT32 && push!(types, Float32)

    for T in types
        @testset "Type $T" begin

            p1 = TuckerPoint(randn(T, n⃗...), r⃗)
            U = (randn(T, n⃗[1], r⃗[1]), randn(T, n⃗[2], r⃗[2]), randn(T, n⃗[3], r⃗[3]))
            p2 = TuckerPoint(randn(T, r⃗), U...)
            p = p2
            U⊥ = map(u -> nullspace(u') * randn(T, size(u, 1) - size(u, 2), size(u, 2)), U)
            v = TuckerTVector(randn(T, r⃗), U⊥)
            U⊥ = map(u -> nullspace(u') * randn(T, size(u, 1) - size(u, 2), size(u, 2)), U)
            w = TuckerTVector(randn(T, r⃗), U⊥)

            @test inner(M, p, v, v) ≈ norm(M, p, v)^2
            @test p ≈ TuckerPoint(p.hosvd.core, p.hosvd.U...)
            @test v == TuckerTVector(v.Ċ, v.U̇)

            @test representation_size(M) == n⃗
            @test manifold_dimension(M) ==
                prod(r⃗) + sum(ntuple(d -> r⃗[d] * (n⃗[d] - r⃗[d]), length(r⃗)))

            @test is_point(M, p1)
            @test is_point(M, p2)
            @test !is_point(M, TuckerPoint(zeros(T, r⃗), U...))
            @test !is_point(M, TuckerPoint(randn(T, n⃗), n⃗))
            @test !is_point(M, TuckerPoint(randn(T, n⃗ .+ 1), r⃗))

            @test is_vector(M, p2, v)
            @test !is_vector(M, p1, v)
            @test !is_vector(M, p2, TuckerTVector(randn(T, r⃗), map(u⊥ -> u⊥[:, 1:(end - 1)], U⊥)))

            @test vec(embed(M, p1)) ≈ kron(reverse(p1.hosvd.U)...) * vec(p1.hosvd.core)
            @test inner(M, p2, v, w) ≈ dot(embed(M, p2, v), embed(M, p2, w))

            pts = [TuckerPoint(randn(n⃗...), r⃗) for i in 1:3]
            test_manifold(
                M,
                pts;
                is_mutating = false, # avoid allocations of the wrong type
                basis_types_to_from = (DefaultOrthonormalBasis(),),
                basis_types_vecs = (DefaultOrthonormalBasis(),),
                test_exp_log=false,
                default_inverse_retraction_method=ProjectionInverseRetraction(),
                test_injectivity_radius=false,
                default_retraction_method=PolarRetraction(),
                test_is_tangent=false,
                test_project_tangent=false,
                test_default_vector_transport=false,
                test_forward_diff=false,
                test_reverse_diff=false,
                test_vector_spaces=false,
                test_vee_hat=false,
                test_tangent_vector_broadcasting=true,
                projection_atol_multiplier=15,
                retraction_methods=[PolarRetraction()],
                inverse_retraction_methods=[ProjectionInverseRetraction()],
                mid_point12=nothing,
            )
            end
        end
end
