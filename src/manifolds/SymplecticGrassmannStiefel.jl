@doc raw"""
    check_point(M::SymplecticGrassmann, p; kwargs...)

Check whether `p` is a valid point on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)`` manifold by verifying that it
is a valid representer of an equivalence class of the corersponding
[`SymplecticStiefel`](@ref) manifold.
"""
function check_point(M::SymplecticGrassmann, p; kwargs...)
    n, k = get_parameter(M.size)
    return check_point(SymplecticStiefel(2 * n, 2 * k), p; kwargs...)
end

@doc raw"""
    check_vector(M::SymplecticGrassmann, p, X; kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)`` manifold by verifying that it
is a valid representer of an equivalence class of the corersponding
[`SymplecticStiefel`](@ref) manifolds tangent space at `p`.
"""
function check_point(M::SymplecticGrassmann, p, X; kwargs...)
    n, k = get_parameter(M.size)
    return check_vector(SymplecticStiefel(2 * n, 2 * k), p, X; kwargs...)
end
