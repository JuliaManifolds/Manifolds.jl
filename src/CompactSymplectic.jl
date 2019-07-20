"""
    QuaternionMultiplicationOperation <: AbstractGroupOperation

Group operation that consists of quaternion multiplication, also known as
the Hamilton product.
"""
struct QuaternionMultiplicationOperation <: AbstractGroupOperation end

@doc doc"""
    CompactSymplectic <: AbstractGroupManifold{QuaternionMultiplicationOperation}

The compact symplectic group $Sp(n)$, which is also the quaternionic unitary
group, is the group of $n \times n$ matrices whose elements are quaternions.
Topologically, $Sp(1)$ is the 3-Sphere. Quaternion elements are represented
with the elements $q=[s, v_1, v_2, v_3]$.

# Constructor

    CompactSymplectic(n)
"""
struct CompactSymplectic{N} <: AbstractGroupManifold{QuaternionMultiplicationOperation} end

CompactSymplectic(n) = CompactSymplectic{n}()

@traitimpl IsDecoratorManifold{CompactSymplectic{1}}

base_manifold(::CompactSymplectic{1}) = Sphere(3)

function show(io::IO, ::CompactSymplectic{N}) where {N}
    print(io, "CompactSymplectic($N)")
end

function hamilton_prod!(z, x, y)
    @assert length(x) == 4 && length(y) == 4 && length(z) == 4

    @inbounds begin
        xs = x[1]
        xv1 = x[2]
        xv2 = x[3]
        xv3 = x[4]
        ys = y[1]
        yv1 = y[2]
        yv2 = y[3]
        yv3 = y[4]

        z[1] = xs * ys - xv1 * yv1 - xv2 * yv2 - xv3 * yv3
        z[2] = xs * yv1 + xv1 * ys + xv2 * yv3 - xv3 * yv2
        z[3] = xs * yv2 - xv1 * yv3 + xv2 * ys + xv3 * yv1
        z[4] = xs * yv3 + xv1 * yv2 - xv2 * yv1 + xv3 * ys
    end

    return z
end

function hamilton_prod(x, y)
    z = similar(x)
    hamilton_prod!(z, x, y)
    return z
end

hamilton_prod(::Identity{G}, x) where {G<:AbstractGroupManifold{QuaternionMultiplicationOperation}} = x
hamilton_prod(x, ::Identity{G}) where {G<:AbstractGroupManifold{QuaternionMultiplicationOperation}} = x
hamilton_prod(e::E, ::E) where {G<:AbstractGroupManifold{QuaternionMultiplicationOperation},E<:Identity{G}} = e

inv(e::Identity{G}) where {G<:AbstractGroupManifold{QuaternionMultiplicationOperation}} = e

function identity(::CompactSymplectic{1}, x)
    e = similar(x, 4)
    vi = SVector{3}(2:4)
    @inbounds begin
        e[1] = 1
        e[vi] .= 0
    end
    return e
end

function inv(::CompactSymplectic{1}, x)
    @assert length(x) == 4
    y = similar(x)
    vi = SVector{3}(2:4)
    @inbounds begin
        y[1] = x[1]
        y[vi] .= -x[vi]
    end
    return y
end

inv(::G, e::Identity{G}) where {G<:CompactSymplectic{1}} = inv(e)

compose(::CompactSymplectic{1}, x, y) = hamilton_prod(x, y)

function translate_diff(::CompactSymplectic{1},
                        x,
                        y,
                        vy,
                        ::Left)
    return hamilton_prod(x, vy)
end

function translate_diff(::CompactSymplectic{1},
                        x,
                        y,
                        vy,
                        ::Right)
    return hamilton_prod(vy, x)
end

@inline inner(::CompactSymplectic{1}, ::Identity, ve, we) = dot(ve, we)

function exp!(::GT, y, ::Identity{GT}, v) where {GT<:CompactSymplectic{1}}
      @assert length(v) == 4 && length(y) == 4
      vi = SVector{3}(2:4)
      @inbounds begin
          θu = v[vi]
          θ = norm(θu)
          y[1] = cos(θ)
          y[vi] .= θu .* usinc(θ)
      end
      return y
end

function log!(::GT, v, ::Identity{GT}, y) where {GT<:CompactSymplectic{1}}
    @assert length(v) == 4 && length(y) == 4
    vi = SVector{3}(2:4)
    @inbounds begin
        sinθv = y[vi]
        θ = atan(norm(sinθv), y[1])
        v[1] = 0
        v[vi] .= sinθv ./ usinc(θ)
    end
    return v
end
