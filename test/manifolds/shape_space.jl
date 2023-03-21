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
    @test !is_flat(M)
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
    X1 = [
        0.6090792159558263 -0.02523987621672985 -0.5838393397390964
        0.4317628895706799 0.12108361184633629 -0.5528465014170161
    ]
    X1h = [
        0.5218590427922166 0.05525104866717821 -0.5771100914593948
        0.3319078589730016 0.2777009756923593 -0.6096088346653609
    ]
    X1v = [
        0.08722017316360964 -0.08049092488390806 -0.006729248279701561
        0.09985503059767825 -0.156617363846023 0.05676233324834479
    ]
    @testset "tangent vector components" begin
        @test isapprox(M, p1, horizontal_component(M, p1, X1), X1h)
        @test isapprox(M, p1, vertical_component(M, p1, X1), X1v)
        Y = similar(X1)
        vertical_component!(M, Y, p1, X1)
        @test isapprox(M, p1, Y, X1v)
        @test norm(M, p1, X1v) < 1e-16
        @test abs(norm(M, p1, X1) - norm(M, p1, X1h)) < 1e-16
    end

    @test_throws ManifoldDomainError is_point(M, [1 0 1; 1 -1 0], true)
    @test_throws ManifoldDomainError is_vector(M, p1, [1 0 1; 1 -1 0], true)

    @testset "exp/distance/norm" begin
        q1 = exp(M, p1, X1)
        @test distance(M, p1, q1) ≈ norm(M, p1, X1)
    end

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
    @testset "degenerate cases" begin
        Md3_2 = KendallsShapeSpace(3, 2)
        Md2_1 = KendallsShapeSpace(2, 1)
        @test manifold_dimension(Md3_2) == 0
        @test manifold_dimension(Md2_1) == 0
    end
end
