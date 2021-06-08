include("utils.jl")

@testset "Tucker" begin
    Random.seed!(1)

    n⃗ = (4, 5, 6)
    r⃗ = (2, 3, 4)
    M = Tucker(n⃗, r⃗)
    p1 = TuckerPoint(randn(n⃗...), r⃗)
    U = (randn(n⃗[1], r⃗[1]), randn(n⃗[2], r⃗[2]), randn(n⃗[3], r⃗[3]))
    p2 = TuckerPoint(randn(r⃗), U...)
    p = p2
    U⊥ = map(u -> nullspace(u') * randn(size(u, 1) - size(u, 2), size(u, 2)), U)
    v = TuckerTVector(randn(r⃗), U⊥)

    @test inner(M, p, v, v) == norm(M, p, v)^2
    @test p ≈ TuckerPoint(p.hosvd.core, p.hosvd.U...)
    @test v == TuckerTVector(v.Ċ, v.U̇)

    @test representation_size(M) == n⃗
    @test manifold_dimension(M) ==
          prod(r⃗) + sum(ntuple(d -> r⃗[d] * (n⃗[d] - r⃗[d]), length(r⃗)))

    @test is_point(M, p1)
    @test is_point(M, p2)
    @test !is_point(M, TuckerPoint(zeros(r⃗), U...))
    @test !is_point(M, TuckerPoint(randn(n⃗), n⃗))
    @test !is_point(M, TuckerPoint(randn(n⃗ .+ 1), r⃗))

    @test is_vector(M, p2, v)
    @test !is_vector(M, p1, v)
    @test !is_vector(M, p2, TuckerTVector(randn(r⃗), map(u⊥ -> u⊥[:, 1:(end - 1)], U⊥)))

    pts = [TuckerPoint(randn(n⃗...), r⃗) for i in 1:3]
    test_manifold(
        M,
        pts;
        is_mutating = false, # avoid allocations of the wrong type
        #basis_types_to_from = (DefaultOrthonormalBasis(),),
        #basis_types_vecs = (DefaultOrthonormalBasis(),),
        test_exp_log=false,
        default_inverse_retraction_method=ProjectionInverseRetraction(),
        test_injectivity_radius=false,
        default_retraction_method=HOSVDRetraction(),
        test_is_tangent=false,
        test_project_tangent=false,
        test_default_vector_transport=false,
        test_forward_diff=false,
        test_reverse_diff=false,
        test_vector_spaces=false,
        test_vee_hat=false,
        test_tangent_vector_broadcasting=false,
        projection_atol_multiplier=15,
        retraction_methods=[HOSVDRetraction()],
        inverse_retraction_methods=[ProjectionInverseRetraction()],
        mid_point12=nothing,
    )
end
