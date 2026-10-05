# the cardiac cycle is defined by the ventricular "driving function", epsilon. The definition of this function is what gives meaning to the event t = 0, namely the onset of ventricular (isovolumic) contraction/systole. Importantly, t = 0 is not aortic valve opening or peak arterial pressure or any other event.
# StructuralIdentifiabiliy.jl will require a driving function whose periodicity comes from a differentiable function (and not "floor"), perhaps a sinusoid of some kind?

"""
The ventricular driving function for the 6-compartment model specified in Burkhoff and Tyberg 1993. 
Inputs: ventricular volume, V at time t, and properties of the time-varying ventricular elastance, including contractile properties (Emax, Tmax) and relaxation properties (A,B,τ). 
Outputs: Pressure in the ventricle at time t. This pressure is periodic and depends on t relative to HR. If timecheck=true the function returns the value of the epsilon driving function at the end of the cardiac cycle. 
"""
# Burkhoff's epsilon driving function: contraction timer
function epsilon_Burkhoff(t,T,τ,HR)
    epsilon(x) = ifelse(x < 3.0T / 2.0, 
        0.5*(sin(pi * x / T - (pi / 2.0)) + 1.0), 
        0.5*exp(((3.0T / 2.0) - x) / τ)
    )
    tcycle = tcycle_from_HR(HR)

    eps = epsilon(t)
    eps_ed = epsilon(tcycle*0.99)

    return (eps=eps,eps_ed=eps_ed)
end

# smoothed version of Burkhoff's epsilon driving function
function epsilon_smooth(t,T,τ,HR,a=0.03,b=3.0)

    L(x) = 1.0 / (1.0 + exp(-a * (x - 3.0*T/2.0))^b)

    epsilon(x) = 
        (1.0-L(x))*0.5*(sin(pi * x / T - (pi / 2.0)) + 1.0) + 
        L(x)*0.5*exp(((3.0*T / 2.0) - x) / τ)

    # The L(x) weighted mixture of the sin wave and exponential leads to a maximum epsilon that is not exactly 1.0 when x=T, and it doesn't quite start at 0. It's easy to remove the latter bias, but rescaling is more challenging. Consider using Symbolics to find an analytic formula for the max and then rescale analytically. Finding the max numerically can be time-consuming, so try to avoid. 

    tcycle = tcycle_from_HR(HR)
    eps = epsilon(t)
    eps_norm = epsilon(t) - epsilon(0.0)
    eps_ed = epsilon(tcycle*0.99)

    return (eps=eps_norm,eps_ed=eps_ed)

end

"""
    smoothstep_quintic(x)

Calculates the quintic smoothstep function 6x^5 - 15x^4 + 10x^3 for x in [0, 1].
Returns 0 for x < 0 and 1 for x > 1. Provides C2 continuity (continuous
first and second derivatives).
"""
function smoothstep_quintic(x::Real)
    # Clamp x to the [0, 1] range before applying the polynomial
    clamped_x = max(0.0, min(1.0, x))
    # Calculate 6x^5 - 15x^4 + 10x^3 using Horner's method for efficiency
    return clamped_x * clamped_x * clamped_x * (10.0 + clamped_x * (-15.0 + clamped_x * 6.0))
end

"""
    epsilon_smooth_2(t, T, τ, HR, delta)

Calculates the smoothed epsilon value at time `t`, blending between a
sine-like function and an exponential decay around t = 1.5*T.

# Arguments
- `t::Real`: Current time.
- `T::Real`: Characteristic time for the sine part (determines period and peak at t=T). Must be positive.
- `τ::Real`: Characteristic time for the exponential decay part. Must be positive.
- `HR::Real`: Heart rate in beats per minute.
- `delta::Real`: Half-width of the smoothing transition interval around 1.5*T.
                 Must be non-negative. Set delta = 0 to recover the original
                 piecewise function with a potential derivative discontinuity.
                 A typical value might be 0.05*T or similar.

# Returns
- `::NamedTuple`: A tuple with fields `eps` and `eps_ed` containing the calculated epsilon values.
"""
function epsilon_smooth_2(t, T, τ, HR, delta=10.0)
    # Define the internal epsilon function
    function epsilon(x)
        # --- Define the two function parts ---
        # Part 1: Sine-based function (0.5 * (1 - cos(pi*t/T)))
        # Valid for t < 1.5*T in the original function.
        # Satisfies f1(0)=0, f1(T)=1, f1(1.5*T)=0.5
        f1(x) = 0.5 * (sin(pi * x / T - (pi / 2.0)) + 1.0)

        # Part 2: Exponential decay
        # Valid for t > 1.5*T in the original function.
        # Satisfies f2(1.5*T)=0.5
        f2(x) = 0.5 * exp(((1.5 * T) - x) / τ)

        # --- Determine Transition Region ---
        t_trans = 1.5 * T
        t1 = t_trans - delta
        t2 = t_trans + delta

        # --- Calculate Epsilon ---
        # Use ifelse instead of if statements for symbolic compatibility
        return ifelse(delta == 0.0,
            # No smoothing, use original piecewise definition
            ifelse(x < t_trans, f1(x), f2(x)),
            # With smoothing
            ifelse(x <= t1,
                # Before the transition interval, use function 1
                f1(x),
                ifelse(x >= t2,
                    # After the transition interval, use function 2
                    f2(x),
                    # Inside the transition interval [t1, t2]
                    # Normalize t to x in [0, 1] for the smoothstep function
                    ifelse(t2 <= t1,
                        # Should only occur if delta is extremely small, effectively zero
                        f1(t_trans), # or f2(t_trans), they are equal here
                        # Calculate the smoothstep weight and blend the functions
                        begin
                            x_norm = (x - t1) / (t2 - t1)
                            s = smoothstep_quintic(x_norm)
                            (1.0 - s) * f1(x) + s * f2(x)
                        end
                    )
                )
            )
        )
    end

    # Calculate the epsilon value at time t
    eps = epsilon(t)
    
    # Calculate end-diastolic value (at t = 0.99 * tcycle)
    tcycle = tcycle_from_HR(HR)
    eps_ed = epsilon(tcycle * 0.99)
    
    return (eps=eps, eps_ed=eps_ed)
end

"""
    epsilon_super(t1, T, τ, HR; n_prior=5, a=0.03, b=3.0) -> (eps, eps_ed)

Twitch-superposition contraction timer — an **opt-in** alternative to
[`epsilon_smooth`](@ref). Instead of resetting the activation to 0 each cycle, it
sums the current beat's activation with the still-decaying tails of the previous
`n_prior` beats, so the driving function starts each cycle from the carried-over
residual and is **continuous across the cycle boundary** (no end-diastolic
discontinuity at high HR — incomplete relaxation is represented smoothly rather
than as a reset artifact).

The summed activation is renormalized so its value at the current-beat peak time
`T` is 1, preserving the identified `Ees ↔ peak-elastance` mapping (the true cycle
max can exceed 1 by a fraction of a percent since the sum peaks slightly before
`T`). At resting HR the prior-beat tails are negligible, so this reduces to
[`epsilon_smooth`](@ref) and the fitted twin is unchanged; the models diverge only
at high HR.

`n_prior` is fixed (not HR-adaptive) because the model is built symbolically with
`HR` symbolic; 5 covers `relax_time/tcycle` through very high rates, and surplus
terms contribute ~0. Use via `BurkhoffElastance4Eed(contraction_timer=epsilon_super)`.
"""
function epsilon_super(t1, T, τ, HR; n_prior::Int = 5, a = 0.03, b = 3.0)
    tcycle = tcycle_from_HR(HR)
    # Single-beat activation for age s ≥ 0 (same kernel as epsilon_smooth): sine
    # rise peaking at 1 at s=T, smoothly blended into a 0.5·exp((3T/2−s)/τ) decay.
    kernel(s) = begin
        L = 1.0 / (1.0 + exp(-a * (s - 1.5 * T))^b)
        (1.0 - L) * 0.5 * (sin(pi * s / T - pi / 2.0) + 1.0) + L * 0.5 * exp((1.5 * T - s) / τ)
    end
    # current beat (j=0) plus the decaying tails of the n_prior previous beats (j≥1)
    summed(φ) = sum(kernel(φ + j * tcycle) for j in 0:n_prior)
    nrm = summed(T)                        # pin peak ≈ 1 at the current-beat peak time
    return (eps = summed(t1) / nrm, eps_ed = summed(0.99 * tcycle) / nrm)
end

# Elastance constructors; they depend on the contraction timer (epsilon_smooth),
# so include this file after utils.jl. Timing consistency is no longer stored
# here — it is the pure function `timing_errors(pall)` in utils.jl.
function BurkhoffElastance4(; 
    contraction_timer::Function = epsilon_smooth)
    return BurkhoffElastance4{typeof(contraction_timer)}(contraction_timer)
end

function BurkhoffElastance4Eed(;
    contraction_timer::Function = epsilon_smooth,
    Vref_lv::Float64 = 140.0, Vref_rv::Float64 = 120.0,
    Vref_la::Float64 = 60.0,  Vref_ra::Float64 = 45.0)
    return BurkhoffElastance4Eed{typeof(contraction_timer)}(
        contraction_timer, Vref_lv, Vref_rv, Vref_la, Vref_ra)
end

function BurkhoffElastance2(; 
    contraction_timer::Function = epsilon_smooth)
    return BurkhoffElastance2{typeof(contraction_timer)}(contraction_timer)
end

# Elastance functors
function (be::BurkhoffElastance2)(t, Tcontract, τ, HR, V, V0, Ees, A, B)
    tcycle = tcycle_from_HR(HR)
    t1 = ((t / tcycle) - floor(t/tcycle)) * tcycle # t is absolute time, t1 is cardiac cycle time
    eps, _ = be.contraction_timer(t1,Tcontract,τ,HR)
    Pes = Ees * (V - V0)
    Ped = A * (exp(B * (V - V0)) - 1.0)
    return (Pes - Ped) * eps + Ped
end

function (be::BurkhoffElastance4)(t, delay, Tcontract, τ, HR, V, V0, Ees, A, B)
    tcycle = tcycle_from_HR(HR)
    tstart = t - delay # delay = 0 indicates atrial contraction; delay = AVD corresponds to ventricular contraction
    t1 = ((tstart / tcycle) - floor(tstart/tcycle)) * tcycle # tstart is absolute time, t1 is cardiac cycle time
    eps, _ = be.contraction_timer(t1,Tcontract,τ,HR)
    Pes = Ees * (V - V0)
    Ped = A * (exp(B * (V - V0)) - 1.0)
    return (Pes - Ped) * eps + Ped
end

function (be::BurkhoffElastance4Eed)(t, delay, Tcontract, τ, HR, V, V0, Ees, Eed_ref, B, Vref)
    tcycle = tcycle_from_HR(HR)
    tstart = t - delay
    t1 = ((tstart / tcycle) - floor(tstart / tcycle)) * tcycle
    eps, _ = be.contraction_timer(t1, Tcontract, τ, HR)
    Pes = Ees * (V - V0)
    Ped = (Eed_ref / (B * exp(B * (Vref - V0)))) * (exp(B * (V - V0)) - 1.0)
    return (Pes - Ped) * eps + Ped
end
