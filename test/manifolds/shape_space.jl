include("../utils.jl")

@testset "KendallsPreShapeSpace" begin
    M = KendallsPreShapeSpace(2, 3)
    @test representation_size(M) === (2, 3)
    @test manifold_dimension(M) == 3
    @test injectivity_radius(M) == pi

    p1 = [
        0.4385117672460505 -0.6877826444042382 0.24927087715818771
        -0.3830259932279294 0.35347460720654283 0.029551386021386548
    ]
    p2 = [
        -0.42693314765896473 -0.3268567431952937 0.7537898908542584
        0.3054740561061169 -0.18962848284149897 -0.11584557326461796
    ]
    p3 = [
        0.3248027612629014 0.440253011955812 -0.7650557732187135
        0.26502337825226757 -0.06175142812400016 -0.20327195012826738
    ]
    @test_throws DomainError is_point(M, [1 0 1; 1 -1 0] / 2, true)
    @test_throws DomainError is_vector(
        M,
        [-1 0 1.0; 0 0 0] / sqrt(2),
        [1.0 0 1; 1 -1 0],
        true,
    )
    test_manifold(
        M,
        [p1, p2, p3];
        is_point_atol_multiplier=1,
        is_tangent_atol_multiplier=1,
        exp_log_atol_multiplier=5,
        test_project_point=true,
        test_project_tangent=true,
        test_rand_point=true,
        test_rand_tvector=true,
        rand_tvector_atol_multiplier=5,
    )
end

@testset "KendallsShapeSpace" begin
    M = KendallsShapeSpace(2, 3)
    @test manifold_dimension(M) == 2
    @test get_total_space(M) === KendallsPreShapeSpace(2, 3)
    p1 = [
        0.4385117672460505 -0.6877826444042382 0.24927087715818771
        -0.3830259932279294 0.35347460720654283 0.029551386021386548
    ]
    p2 = [
        -0.42693314765896473 -0.3268567431952937 0.7537898908542584
        0.3054740561061169 -0.18962848284149897 -0.11584557326461796
    ]
    p3 = [
        0.3248027612629014 0.440253011955812 -0.7650557732187135
        0.26502337825226757 -0.06175142812400016 -0.20327195012826738
    ]
    @test_throws ManifoldDomainError is_point(M, [1 0 1; 1 -1 0], true)
    @test_throws ManifoldDomainError is_vector(M, p1, [1 0 1; 1 -1 0], true)
    test_manifold(
        M,
        [p1, p2, p3];
        is_point_atol_multiplier=1,
        is_tangent_atol_multiplier=1,
        exp_log_atol_multiplier=2e8,
        projection_atol_multiplier=1,
        test_project_point=true,
        test_project_tangent=true,
        test_rand_point=true,
        test_rand_tvector=true,
        rand_tvector_atol_multiplier=5,
    )
end
