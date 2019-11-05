using Manifolds
using BenchmarkTools
using StaticArrays
using HybridArrays
using LinearAlgebra
using Random

# Define a parent BenchmarkGroup to contain our suite
SUITE = BenchmarkGroup()

Random.seed!(12334)

function add_manifold(M::Manifold, pts, name;
    test_tangent_vector_broadcasting = true,
    retraction_methods = [],
    inverse_retraction_methods = [],
    point_distributions = [],
    tvector_distributions = [])

    SUITE["manifolds"][name] = BenchmarkGroup()
    tv = log(M, pts[1], pts[2])
    tv1 = log(M, pts[1], pts[2])
    tv2 = log(M, pts[1], pts[3])
    p = similar(pts[1])
    SUITE["manifolds"][name]["similar"] = @benchmarkable similar($(pts[1]))
    SUITE["manifolds"][name]["log"] = @benchmarkable log($M, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["log!"] = @benchmarkable log!($M, $tv, $(pts[1]), $(pts[2]))
    for iretr ∈ inverse_retraction_methods
        SUITE["manifolds"][name]["inverse_retract: "*string(iretr)] = @benchmarkable inverse_retract($M, $(pts[1]), $(pts[2]), $iretr)
        SUITE["manifolds"][name]["inverse_retract!: "*string(iretr)] = @benchmarkable inverse_retract!($M, $tv, $(pts[1]), $(pts[2]), $iretr)
    end
    SUITE["manifolds"][name]["exp"] = @benchmarkable exp($M, $(pts[1]), $tv1)
    SUITE["manifolds"][name]["exp!"] = @benchmarkable exp!($M, $p, $(pts[1]), $tv1)
    for retr ∈ retraction_methods
        SUITE["manifolds"][name]["retract: "*string(retr)] = @benchmarkable retract($M, $(pts[1]), $tv1, $retr)
        SUITE["manifolds"][name]["retract!: "*string(retr)] = @benchmarkable retract!($M, $p, $(pts[1]), $tv1, $retr)
    end
    SUITE["manifolds"][name]["norm"] = @benchmarkable norm($M, $(pts[1]), $tv1)
    SUITE["manifolds"][name]["inner"] = @benchmarkable inner($M, $(pts[1]), $tv1, $tv2)
    SUITE["manifolds"][name]["distance"] = @benchmarkable distance($M, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["isapprox (pt)"] = @benchmarkable isapprox($M, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["isapprox (tv)"] = @benchmarkable isapprox($M, $(pts[1]), $tv1, $tv2)
    SUITE["manifolds"][name]["2 * tv1 + 3 * tv2"] = @benchmarkable 2 * $tv1 + 3 * $tv2
    SUITE["manifolds"][name]["tv = 2 * tv1 + 3 * tv2"] = @benchmarkable $tv = 2 * $tv1 + 3 * $tv2
    if test_tangent_vector_broadcasting
        SUITE["manifolds"][name]["tv = 2 .* tv1 .+ 3 .* tv2"] = @benchmarkable $tv = 2 .* $tv1 .+ 3 .* $tv2
        SUITE["manifolds"][name]["tv .= 2 .* tv1 .+ 3 .* tv2"] = @benchmarkable $tv .= 2 .* $tv1 .+ 3 .* $tv2
    end
    for pd ∈ point_distributions
        distr_name = string(pd)
        distr_name = distr_name[1:min(length(distr_name), 50)]
        SUITE["manifolds"][name]["point distribution "*distr_name] = @benchmarkable rand($pd)
    end
    for tvd ∈ point_distributions
        distr_name = string(tvd)
        distr_name = distr_name[1:min(length(distr_name), 50)]
        SUITE["manifolds"][name]["tangent vector distribution "*distr_name] = @benchmarkable rand($tvd)
    end
end

# General manifold benchmarks
function add_manifold_benchmarks()

    SUITE["manifolds"] = BenchmarkGroup()

    s2 = Manifolds.Sphere(2)
    array_s2 = ArrayManifold(s2)
    r2 = Manifolds.Euclidean(2)

    pts_r2 = [Size(2)([1.0, 1.0]),
              Size(2)([-2.0, 3.0]),
              Size(2)([3.0, -2.0])]

    add_manifold(r2,
                 pts_r2,
                 "Euclidean{2} -- SizedArray")

    add_manifold(r2,
                 [MVector{2,Float64}([1.0, 1.0]),
                  MVector{2,Float64}([-2.0, 3.0]),
                  MVector{2,Float64}([3.0, -2.0])],
                  "Euclidean{2} -- MVector")

    ud_sphere = Manifolds.uniform_distribution(s2, Size(3)([1.0, 0.0, 0.0]))
    gtd_sphere = Manifolds.normal_tvector_distribution(s2, Size(3)([1.0, 0.0, 0.0]), 1.0)

    pts_s2 = [Size(3)([1.0, 0.0, 0.0]),
              Size(3)([0.0, 1.0, 0.0]),
              Size(3)([0.0, 0.0, 1.0])]

    add_manifold(s2,
                 pts_s2,
                 "Sphere{2} -- SizedArray";
                 point_distributions = [ud_sphere],
                 tvector_distributions = [gtd_sphere])

    add_manifold(array_s2,
                 [Size(3)([1.0, 0.0, 0.0]),
                  Size(3)([0.0, 1.0, 0.0]),
                  Size(3)([0.0, 0.0, 1.0])],
                 "ArrayManifold{Sphere{2}} -- SizedArray";
                 test_tangent_vector_broadcasting = false)

    retraction_methods_rot = [Manifolds.PolarRetraction(),
                              Manifolds.QRRetraction()]

    inverse_retraction_methods_rot = [Manifolds.PolarInverseRetraction(),
                                      Manifolds.QRInverseRetraction()]

    so2 = Manifolds.Rotations(2)
    angles = (0.0, π/2, 2π/3)
    add_manifold(so2,
                 [Size(2, 2)([cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles],
                 "Rotations(2) -- SizedArray",
                 retraction_methods = retraction_methods_rot,
                 inverse_retraction_methods = inverse_retraction_methods_rot)

    so3 = Manifolds.Rotations(3)
    angles = (0.0, π/2, 2π/3)
    add_manifold(so3,
                 [Size(3, 3)([cos(ϕ) -sin(ϕ) 0; sin(ϕ) cos(ϕ) 0; 0 0 1]) for ϕ in angles],
                 "Rotations(3) -- SizedArray",
                 retraction_methods = retraction_methods_rot,
                 inverse_retraction_methods = inverse_retraction_methods_rot)


    m_prod = Manifolds.ProductManifold(s2, r2)
    shape_s2r2_array = Manifolds.ShapeSpecification(Manifolds.ArrayReshaper(), s2, r2)
    shape_s2r2_static = Manifolds.ShapeSpecification(Manifolds.StaticReshaper(), s2, r2)

    pts_prd_base = [[1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.1]]
    pts_prod_static = map(p -> Manifolds.ProductArray(shape_s2r2_static, p), pts_prd_base)
    add_manifold(m_prod, pts_prod_static, "ProductManifold with ProductArray (static)")

    pts_prod_array = map(p -> Manifolds.ProductArray(shape_s2r2_array, p), pts_prd_base)
    add_manifold(m_prod, pts_prod_array, "ProductManifold with ProductArray (array)")

    pts_prod_mpoints = [Manifolds.ProductRepr(p[1], p[2]) for p in zip(pts_s2, pts_r2)]
    add_manifold(m_prod, pts_prod_mpoints, "ProductManifold with MPoint";
        test_tangent_vector_broadcasting = false)

    # vector spaces and bundles
    begin
        T = MVector{3, Float64}
        TB = TangentBundle(s2)

        pts_tb = [ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, -1.0, -1.0])),
                  ProductRepr(convert(T, [0.0, 1.0, 0.0]), convert(T, [2.0, 0.0, 1.0])),
                  ProductRepr(convert(T, [1.0, 0.0, 0.0]), convert(T, [0.0, 2.0, -1.0]))]
        add_manifold(TB, pts_tb, "Tangent bundle of S² using MVectors, ProductRepr";
            test_tangent_vector_broadcasting = false)
    end

    # power manifolds
    begin
        Ms = Sphere(2)
        Ms1 = PowerManifold(Ms, 5)
        Ms2 = PowerManifold(Ms, 5, 7)
        Mr = Manifolds.Rotations(3)
        Mr1 = PowerManifold(Mr, 5)
        Mr2 = PowerManifold(Mr, 5, 7)

        types_s1 = [Array{Float64,2},
                    HybridArray{Tuple{3,StaticArrays.Dynamic()}, Float64, 2}]
        types_s2 = [Array{Float64,3},
                    HybridArray{Tuple{3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Float64, 3}]

        types_r1 = [Array{Float64,3},
                    HybridArray{Tuple{3,3,StaticArrays.Dynamic()}, Float64, 3}]
        types_r2 = [Array{Float64,4},
                    HybridArray{Tuple{3,3,StaticArrays.Dynamic(),StaticArrays.Dynamic()}, Float64, 4}]

        retraction_methods = [Manifolds.PowerRetraction(Manifolds.ExponentialRetraction())]
        inverse_retraction_methods = [Manifolds.InversePowerRetraction(Manifolds.LogarithmicInverseRetraction())]

        sphere_dist = Manifolds.uniform_distribution(Ms, @SVector [1.0, 0.0, 0.0])
        power_s1_pt_dist = Manifolds.PowerPointDistribution(Ms1, sphere_dist, randn(Float64, 3, 5))
        power_s2_pt_dist = Manifolds.PowerPointDistribution(Ms2, sphere_dist, randn(Float64, 3, 5, 7))
        sphere_tv_dist = Manifolds.normal_tvector_distribution(Ms, (@MVector [1.0, 0.0, 0.0]), 1.0)
        power_s1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms1), rand(power_s1_pt_dist), sphere_tv_dist)
        power_s2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Ms2), rand(power_s2_pt_dist), sphere_tv_dist)

        id_rot = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        rotations_dist = Manifolds.normal_rotation_distribution(Mr, id_rot, 1.0)
        power_r1_pt_dist = Manifolds.PowerPointDistribution(Mr1, rotations_dist, randn(Float64, 3, 3, 5))
        power_r2_pt_dist = Manifolds.PowerPointDistribution(Mr2, rotations_dist, randn(Float64, 3, 3, 5, 7))
        rotations_tv_dist = Manifolds.normal_tvector_distribution(Mr, MMatrix(id_rot), 1.0)
        power_r1_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mr1), rand(power_r1_pt_dist), rotations_tv_dist)
        power_r2_tv_dist = Manifolds.PowerFVectorDistribution(TangentBundleFibers(Mr2), rand(power_r2_pt_dist), rotations_tv_dist)

        trim(s::String) = s[1:min(length(s), 20)]

        for T in types_s1
            pts1 = [convert(T, rand(power_s1_pt_dist)) for _ in 1:3]
            add_manifold(Ms1, pts1, "power manifold S²^(5,), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true)
        end
        for T in types_s2
            pts2 = [convert(T, rand(power_s2_pt_dist)) for _ in 1:3]
            add_manifold(Ms2, pts2, "power manifold S²^(5,7), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true)
        end

        for T in types_r1
            pts1 = [convert(T, rand(power_r1_pt_dist)) for _ in 1:3]
            add_manifold(Mr1, pts1, "power manifold SO(3)^(5,), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true)
        end
        for T in types_r2
            pts2 = [convert(T, rand(power_r2_pt_dist)) for _ in 1:3]
            add_manifold(Mr2, pts2, "power manifold SO(3)^(5,7), type $(trim(string(T)))";
                test_tangent_vector_broadcasting = true)
        end

    end

end

add_manifold_benchmarks()
