
const CotangentBundle{𝔽,M} =
    VectorBundle{𝔽,CotangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

CotangentBundle(M::AbstractManifold) = VectorBundle(CotangentSpace, M)
function CotangentBundle(M::AbstractManifold, vtm::FiberBundleProductVectorTransport)
    return VectorBundle(CotangentSpace, M, vtm)
end

"""
    const VectorBundleVectorTransport = FiberBundleProductVectorTransport

Deprecated: an alias for `FiberBundleProductVectorTransport`.
"""
const VectorBundleVectorTransport = FiberBundleProductVectorTransport

const CotangentBundle{𝔽,M} =
    VectorBundle{𝔽,CotangentSpaceType,M} where {𝔽,M<:AbstractManifold{𝔽}}

function representation_size(B::CotangentBundleFibers)
    return representation_size(B.manifold)
end

Base.show(io::IO, vb::CotangentBundle) = print(io, "CotangentBundle($(vb.manifold))")

"""
    fiber_bundle_transport(fiber::FiberType, M::AbstractManifold)

Determine the vector tranport used for [`exp`](@ref exp(::FiberBundle, ::Any...)) and
[`log`](@ref log(::FiberBundle, ::Any...)) maps on a vector bundle with fiber type
`fiber` and manifold `M`.
"""
fiber_bundle_transport(::VectorSpaceType, ::AbstractManifold) = ParallelTransport()
