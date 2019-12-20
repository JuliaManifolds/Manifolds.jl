push!(_adbackends, :forwarddiff)

_gradient(f, x::Number, ::Val{:forwarddiff}) = ForwardDiff.derivative(f, x)
_gradient(f, x, ::Val{:forwarddiff}) = ForwardDiff.gradient(f, x)

_jacobian(f, x, ::Val{:forwarddiff}) = ForwardDiff.jacobian(f, x)
