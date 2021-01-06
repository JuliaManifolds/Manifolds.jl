include("utils.jl")

@testset "Essential manifold" begin
    M = EssentialManifold()
    a = π / 6
    b = π / 6
    c = π / 6
    r1 = [1.0 0.0 0.0; 0.0 cos(a) -sin(a); 0.0 sin(a) cos(a)]
    r2 = [cos(b) 0.0 sin(b); 0.0 1.0 0.0; -sin(b) 0.0 cos(b)]
    r3 = [cos(c) -sin(c) 0.0; sin(c) cos(c) 0.0; 0.0 0.0 1.0]
    nr = [1.0 0.0 0.0; 0.0 0.0 -1.0; 0.0 -1.0 0.0]
    p1 = [r1, r2]
    p2 = [r1, r3]
    p3 = [r2, r2]
    @testset "Essential manifold Basics" begin
        @test M.manifold == Rotations(3)
        @test repr(M) == "EssentialManifold(true)"
        @test representation_size(M) == (3, 3, 2)
        @test manifold_dimension(M) == 5
        np1 = [r1, nr]
        np2 = [nr, nr]
        np3 = [r1, r2, r3]
        @test !is_manifold_point(M, r1)
        @test_throws DomainError is_manifold_point(M, r1, true)
        @test_throws DomainError is_manifold_point(M, np3, true)
        @test is_manifold_point(M, p1)
        @test_throws ComponentManifoldError is_manifold_point(M, np1, true)
        @test_throws CompositeManifoldError is_manifold_point(M, np2, true)
        @test !is_tangent_vector(M, p1, 0.0)
        @test_throws DomainError is_tangent_vector(
            M,
            p1,
            [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
            true,
        )
        @test !is_tangent_vector(M, np1, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
        @test !is_tangent_vector(M, p1, p2)
        # projection test
        @test is_tangent_vector(M, p1, project(M, p1, log(M, p1, p2)))
        @test is_tangent_vector(M, p1, project(M, p1, log(M, p2, p1)))
    end
    @testset "Signed Essential" begin
        test_manifold(
            M,
            [p1, p2, p3],
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vector_spaces=true,
            test_project_point=true,
            projection_atol_multiplier=10,
            test_project_tangent=false, # since it includes vert_proj.
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            test_representation_size=true,
            test_exp_log=true,
            mid_point12=nothing,
            exp_log_atol_multiplier=4,
        )
    end
    @testset "Unsigned Essential" begin
        test_manifold(
            EssentialManifold(false),
            [p1, p2, p3],
            test_forward_diff=false,
            test_reverse_diff=false,
            test_vector_spaces=true,
            test_project_point=true,
            projection_atol_multiplier=10,
            test_project_tangent=false, # since it includes vert_proj.
            test_musical_isomorphisms=false,
            test_default_vector_transport=true,
            test_representation_size=true,
            test_exp_log=true,
            mid_point12=nothing,
            exp_log_atol_multiplier=4,
        )
    end
end
