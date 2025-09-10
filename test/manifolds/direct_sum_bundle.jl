using Test
using Manifolds
using RecursiveArrayTools

@testset "Multitangent bundle" begin
    M = Sphere(2)
    m_prod_retr = Manifolds.FiberBundleProductRetraction()
    m_prod_invretr = Manifolds.FiberBundleInverseProductRetraction()
    m_sasaki = SasakiRetraction(5)

    @testset "Nice access to vector bundle components" begin
        TB = Manifolds.MultitangentBundle{2}(M)
        p = ArrayPartition(
            [1.0, 0.0, 0.0],
            ArrayPartition([0.0, 2.0, 4.0], [0.0, -1.0, 0.0]),
        )
        @test p[TB, :point] === p.x[1]
        p[TB, :point] = [0.0, 1.0, 0.0]
        @test p.x[1] == [0.0, 1.0, 0.0]
        @test_throws DomainError p[TB, :error]
        @test_throws DomainError p[TB, :error] = [1, 2, 3]

        @test view(p, TB, :point) === p.x[1]
        view(p, TB, :point) .= [2.0, 3.0, 5.0]
        @test p.x[1] == [2.0, 3.0, 5.0]
        @test_throws DomainError view(p, TB, :error)
    end

    p = [1.0, 0.0, 0.0]
    TB = Manifolds.MultitangentBundle{2}(M)
    # @test sprint(show, TB) == "TangentBundle(Sphere(2, ℝ))"
    @test base_manifold(TB) == M
    @test manifold_dimension(TB) == 3 * manifold_dimension(M)
    @test representation_size(TB) === nothing
    @test default_inverse_retraction_method(TB) === m_prod_invretr
    @test default_retraction_method(TB) == m_prod_retr
    @test default_vector_transport_method(TB) isa
        Manifolds.FiberBundleProductVectorTransport

    @testset "Type" begin
        pts_tb = [
            ArrayPartition(
                [1.0, 0.0, 0.0],
                ArrayPartition([0.0, -1.0, -1.0], [0.0, -1.0, -1.0]),
            ),
            ArrayPartition(
                [0.0, 1.0, 0.0],
                ArrayPartition([2.0, 0.0, 1.0], [-1.0, 0.0, -2.0]),
            ),
            ArrayPartition(
                [1.0, 0.0, 0.0],
                ArrayPartition([0.0, 2.0, -1.0], [0.0, -2.0, -1.0]),
            ),
        ]

        for pt in pts_tb
            @test bundle_projection(TB, pt) ≈ pt.x[1]
        end
        X12_prod = inverse_retract(TB, pts_tb[1], pts_tb[2], m_prod_invretr)
        X13_prod = inverse_retract(TB, pts_tb[1], pts_tb[3], m_prod_invretr)
        basis_types =
            (DefaultOrthonormalBasis(), get_basis(TB, pts_tb[1], DefaultOrthonormalBasis()))
        test_manifold(
            TB,
            pts_tb,
            default_inverse_retraction_method = m_prod_invretr,
            default_retraction_method = m_prod_retr,
            inverse_retraction_methods = [m_prod_invretr],
            retraction_methods = [m_prod_retr],
            test_exp_log = false,
            test_injectivity_radius = false,
            test_tangent_vector_broadcasting = false,
            test_default_vector_transport = true,
            vector_transport_methods = [],
            basis_types_vecs = basis_types,
            projection_atol_multiplier = 4,
            test_inplace = true,
            test_representation_size = false,
            test_rand_point = true,
            test_rand_tvector = true,
        )

        Xir = allocate(pts_tb[1])
        inverse_retract!(TB, Xir, pts_tb[1], pts_tb[2], m_prod_invretr)
        @test isapprox(TB, pts_tb[1], Xir, X12_prod)
        @test isapprox(
            norm(TB.fiber, pts_tb[1][TB, :point], pts_tb[1][TB, :vector]),
            sqrt(
                inner(
                    TB.fiber,
                    pts_tb[1][TB, :point],
                    pts_tb[1][TB, :vector],
                    pts_tb[1][TB, :vector],
                ),
            ),
        )
        @test isapprox(
            distance(
                TB.fiber,
                pts_tb[1][TB, :point],
                pts_tb[1][TB, :vector],
                ArrayPartition([0.0, 2.0, 3.0], [0.0, 2.0, 2.0]),
            ),
            9.273618495495704,
        )
        Xir2 = allocate(pts_tb[1])
        vector_transport_to!(
            TB,
            Xir2,
            pts_tb[1],
            Xir,
            pts_tb[2],
            Manifolds.FiberBundleProductVectorTransport(),
        )
        @test is_vector(TB, pts_tb[2], Xir2)
    end

    @test base_manifold(TangentBundle(M)) == M
end
