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
    inverse_retraction_methods = [])

    SUITE["manifolds"][name] = BenchmarkGroup()
    tv = log(M, pts[1], pts[2])
    tv1 = log(M, pts[1], pts[2])
    tv2 = log(M, pts[1], pts[3])
    p = similar(pts[1])
    SUITE["manifolds"][name]["similar"] = @benchmarkable similar($(pts[1]))
    SUITE["manifolds"][name]["log"] = @benchmarkable log($M, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["log!"] = @benchmarkable log!($M, $tv, $(pts[1]), $(pts[2]))
    SUITE["manifolds"][name]["exp"] = @benchmarkable exp($M, $(pts[1]), $tv1)
    SUITE["manifolds"][name]["exp"] = @benchmarkable exp!($M, $p, $(pts[1]), $tv1)
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

    add_manifold(s2,
                 [Size(3)([1.0, 0.0, 0.0]),
                  Size(3)([0.0, 1.0, 0.0]),
                  Size(3)([0.0, 0.0, 1.0])],
                  "Sphere{2} -- SizedArray")

    add_manifold(array_s2,
                 [Size(3)([1.0, 0.0, 0.0]),
                  Size(3)([0.0, 1.0, 0.0]),
                  Size(3)([0.0, 0.0, 1.0])],
                  "ArrayManifold{Sphere{2}} -- SizedArray";
                  test_tangent_vector_broadcasting = false)

    so2 = Manifolds.Rotations(2)
    angles = (0.0, π/2, 2π/3)
    add_manifold(so2,
                 [Size(2, 2)([cos(ϕ) sin(ϕ); -sin(ϕ) cos(ϕ)]) for ϕ in angles],
                  "Rotations(2) -- SizedArray")

    m_prod = Manifolds.ProductManifold(s2, r2)

    pts_prd_base = [[1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.1]]
    pts_prod = map(p -> Manifolds.ProductArray(m_prod, p), pts_prd_base)

    add_manifold(m_prod, pts_prod, "ProductManifold")
end

add_manifold_benchmarks()
