using Manifolds
using BenchmarkTools
using StaticArrays
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

    add_manifold(s2,
                 [Size(2)([1.0, 1.0]),
                 Size(2)([-2.0, 3.0]),
                 Size(2)([3.0, -2.0])],
                 "Euclidean{2} -- SizedArray")

    add_manifold(r2,
                 [MVector{2,Float64}([1.0, 1.0]),
                  MVector{2,Float64}([-2.0, 3.0]),
                  MVector{2,Float64}([3.0, -2.0])],
                  "Euclidean{2} -- MVector")

    ud_sphere = Manifolds.uniform_distribution(s2, Size(3)([1.0, 0.0, 0.0]))
    gtd_sphere = Manifolds.normal_tvector_distribution(s2, Size(3)([1.0, 0.0, 0.0]), 1.0)

    add_manifold(s2,
                 [Size(3)([1.0, 0.0, 0.0]),
                  Size(3)([0.0, 1.0, 0.0]),
                  Size(3)([0.0, 0.0, 1.0])],
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
                 [Size(2, 2)([cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]) for ϕ in angles],
                 "Rotations(2) -- SizedArray",
                 retraction_methods = retraction_methods_rot,
                 inverse_retraction_methods = inverse_retraction_methods_rot)

    m_prod = Manifolds.ProductManifold(s2, r2)
    shape_s2r2 = Manifolds.ShapeSpecification(s2, r2)

    pts_prd_base = [[1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.1]]
    pts_prod = map(p -> Manifolds.ProductArray(shape_s2r2, p), pts_prd_base)

    add_manifold(m_prod, pts_prod, "ProductManifold")
end

add_manifold_benchmarks()
