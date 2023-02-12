"""
  prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
function prevfloat_tdir(x, x0, x1)
    x1 > x0 ? prevfloat(x) : nextfloat(x)
end

function nextfloat_tdir(x, x0, x1)
    x1 > x0 ? nextfloat(x) : prevfloat(x)
end

function max_tdir(a, b, x0, x1)
    x1 > x0 ? max(a, b) : min(a, b)
end

alg_autodiff(alg::AbstractNewtonAlgorithm{CS, AD, FDT}) where {CS, AD, FDT} = AD
diff_type(alg::AbstractNewtonAlgorithm{CS, AD, FDT}) where {CS, AD, FDT} = FDT

"""
  value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end
value_derivative(f::F, x::AbstractArray) where {F} = f(x), ForwardDiff.jacobian(f, x)

value(x) = x
value(x::Dual) = ForwardDiff.value(x)
value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

function init_J(x; batch = false)
    x_ = batch ? x[:, 1] : x

    J = ArrayInterfaceCore.zeromatrix(x_)
    if ismutable(x_)
        J[diagind(J)] .= one(eltype(x_))
    else
        J += I
    end

    return batch ? repeat(J, 1, 1, size(x, 2)) : J
end

function dogleg_method(H, g, Δ)
    # Compute the Newton step.
    δN = -H \ g
    # Test if the full step is within the trust region.
    if norm(δN) ≤ Δ
        return δN
    end

    # Calcualte Cauchy point, optimum along the steepest descent direction.
    δsd = -g
    norm_δsd = norm(δsd)
    if norm_δsd ≥ Δ
        return δsd .* Δ / norm_δsd
    end

    # Find the intersection point on the boundary.
    δN_δsd = δN - δsd
    dot_δN_δsd = dot(δN_δsd, δN_δsd)
    dot_δsd_δN_δsd = dot(δsd, δN_δsd)
    dot_δsd = dot(δsd, δsd)
    fact = dot_δsd_δN_δsd^2 - dot_δN_δsd * (dot_δsd - Δ^2)
    tau = (-dot_δsd_δN_δsd + sqrt(fact)) / dot_δN_δsd
    return δsd + tau * δN_δsd
end

_batched_mul(x, y, batch) = x * y
function _batched_mul(x::AbstractArray{T, 3}, y::AbstractMatrix, batch) where {T}
    !batch && return x * y
    return dropdims(batched_mul(x, reshape(y, size(y, 1), 1, size(y, 2))); dims = 2)
end
function _batched_mul(x::AbstractMatrix, y::AbstractArray{T, 3}, batch) where {T}
    !batch && return x * y
    return batched_mul(reshape(x, size(x, 1), 1, size(x, 2)), y)
end
function _batched_mul(x::AbstractArray{T1, 3}, y::AbstractArray{T2, 3},
                      batch) where {T1, T2}
    !batch && return x * y
    return batched_mul(x, y)
end