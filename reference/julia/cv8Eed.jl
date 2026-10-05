

"""
    u0_cv8Eed(pall::NamedTuple) -> NamedTuple

Initial conditions for the CV8Eed model (Eed_ref/B parameterization).
Estimates mean-circulatory-filling-pressure volumes using a linearized passive
compliance and enforces total stressed volume Vs via Vvs.

In the Eed_ref parameterization, A = Eed_ref / (B * exp(B*(Vref-V0))), so:
  - linearized compliance at V0: C = 1/(A*B) = exp(B*(Vref-V0)) / Eed_ref
  - inverse passive PV:  V = V0 + log1p(P * B * exp(B*(Vref-V0)) / Eed_ref) / B
"""
function u0_cv8Eed(pall::NamedTuple)
    @unpack Vs, Cas, Cvs, Cvp, Eap = pall
    @unpack Eedref_lv, Blv, Eedref_rv, Brv, Eedref_la, Bla, Eedref_ra, Bra = pall
    @unpack V0_lv, V0_rv, V0_la, V0_ra = pall
    @unpack Vref_lv, Vref_rv, Vref_la, Vref_ra = pall

    Eap_eff = max(Eap, 1e-6)
    Eedref_lv_eff = max(Eedref_lv, 1e-6)
    Blv_eff = max(Blv, 1e-6)
    Eedref_rv_eff = max(Eedref_rv, 1e-6)
    Brv_eff = max(Brv, 1e-6)
    Eedref_la_eff = max(Eedref_la, 1e-6)
    Bla_eff = max(Bla, 1e-6)
    Eedref_ra_eff = max(Eedref_ra, 1e-6)
    Bra_eff = max(Bra, 1e-6)

    C_lv = exp(Blv_eff * (Vref_lv - V0_lv)) / Eedref_lv_eff
    C_rv = exp(Brv_eff * (Vref_rv - V0_rv)) / Eedref_rv_eff
    C_la = exp(Bla_eff * (Vref_la - V0_la)) / Eedref_la_eff
    C_ra = exp(Bra_eff * (Vref_ra - V0_ra)) / Eedref_ra_eff

    C_total = Cas + Cvs + Cvp + (1.0 / Eap_eff) + C_lv + C_rv + C_la + C_ra
    Pmc = Vs / C_total

    Vas = Cas * Pmc
    Vvs = Cvs * Pmc
    Vvp = Cvp * Pmc
    Vap = Pmc / Eap_eff

    # Inverse passive PV: V = V0 + log1p(P * B * exp(B*(Vref-V0)) / Eed_ref) / B
    Vlv = V0_lv + log1p(Pmc * Blv_eff * exp(Blv_eff * (Vref_lv - V0_lv)) / Eedref_lv_eff) / Blv_eff
    Vrv = V0_rv + log1p(Pmc * Brv_eff * exp(Brv_eff * (Vref_rv - V0_rv)) / Eedref_rv_eff) / Brv_eff
    Vla = V0_la + log1p(Pmc * Bla_eff * exp(Bla_eff * (Vref_la - V0_la)) / Eedref_la_eff) / Bla_eff
    Vra = V0_ra + log1p(Pmc * Bra_eff * exp(Bra_eff * (Vref_ra - V0_ra)) / Eedref_ra_eff) / Bra_eff

    stressed_no_vvs = (Vlv - V0_lv) + (Vrv - V0_rv) + (Vla - V0_la) + (Vra - V0_ra) +
                      Vas + Vap + Vvp
    Vvs = Vs - stressed_no_vvs

    min_Vvs = 50.0
    if Vvs < min_Vvs
        target_stressed = max(Vs - min_Vvs, 0.0)
        scale = stressed_no_vvs > 0 ? (target_stressed / stressed_no_vvs) : 0.0
        Vlv = V0_lv + (Vlv - V0_lv) * scale
        Vrv = V0_rv + (Vrv - V0_rv) * scale
        Vla = V0_la + (Vla - V0_la) * scale
        Vra = V0_ra + (Vra - V0_ra) * scale
        Vas *= scale
        Vap *= scale
        Vvp *= scale
        Vvs = min_Vvs
    end

    return (Vlv=Vlv, Vrv=Vrv, Vas=Vas, Vvs=Vvs, Vra=Vra, Vap=Vap, Vvp=Vvp, Vla=Vla)
end

u0_from_params(::CV8Eed, pall::NamedTuple) = u0_cv8Eed(pall)

function cv8Eed(driving_pressure::BurkhoffElastance4Eed=BurkhoffElastance4Eed())::CV8Eed
    @parameters Emax_LV Emax_RV Tmax τ Eedref_lv Eedref_rv Blv Brv
    @parameters Emax_RA Emax_LA Tmax_a τ_a Eedref_la Eedref_ra Bla Bra
    @parameters AVD HR
    @parameters Cas Cvs Ras
    @parameters Eap Cvp Rap
    @parameters Rmv Rcs Rtv Rcp Rra Rla
    @parameters Vs V0_lv V0_rv V0_la V0_ra
    @parameters Vref_lv Vref_rv Vref_la Vref_ra

    @independent_variables t
    @variables Vlv(t) Qlv(t) Plv(t) Vrv(t) Qrv(t) Prv(t) Vla(t) Qla(t) Pla(t) Vra(t) Qra(t) Pra(t) Vas(t) Qas(t) Pas(t) Vvs(t) Qvs(t) Pvs(t) Vap(t) Qap(t) Pap(t) Vvp(t) Qvp(t) Pvp(t) mv(t) av(t) tv(t) pv(t)

    D = Differential(t)

    eqs = [
        mv ~ ifelse(Pla > Plv, 1.0, 0.0),
        av ~ ifelse(Plv > Pas, 1.0, 0.0),
        tv ~ ifelse(Pra > Prv, 1.0, 0.0),
        pv ~ ifelse(Prv > Pap, 1.0, 0.0),
        Plv ~ driving_pressure(t, AVD, Tmax, τ, HR, Vlv, V0_lv, Emax_LV, Eedref_lv, Blv, Vref_lv),
        Prv ~ driving_pressure(t, AVD, Tmax, τ, HR, Vrv, V0_rv, Emax_RV, Eedref_rv, Brv, Vref_rv),
        Pla ~ driving_pressure(t, 0.0, Tmax_a, τ_a, HR, Vla, V0_la, Emax_LA, Eedref_la, Bla, Vref_la),
        Pra ~ driving_pressure(t, 0.0, Tmax_a, τ_a, HR, Vra, V0_ra, Emax_RA, Eedref_ra, Bra, Vref_ra),
        Pas ~ Vas / Cas,
        Pap ~ Vap * Eap,
        Pvs ~ Vvs / Cvs,
        Pvp ~ Vvp / Cvp,
        Qlv ~ mv * (Pla - Plv) / Rmv,
        Qas ~ av * (Plv - Pas) / Rcs,
        Qvs ~ (Pas - Pvs) / Ras,
        Qra ~ (Pvs - Pra) / Rra,
        Qrv ~ tv * (Pra - Prv) / Rtv,
        Qap ~ pv * (Prv - Pap) / Rcp,
        Qvp ~ (Pap - Pvp) / Rap,
        Qla ~ (Pvp - Pla) / Rla,
        D(Vlv) ~ Qlv - Qas,
        D(Vas) ~ Qas - Qvs,
        D(Vvs) ~ Qvs - Qra,
        D(Vra) ~ Qra - Qrv,
        D(Vrv) ~ Qrv - Qap,
        D(Vap) ~ Qap - Qvp,
        D(Vvp) ~ Qvp - Qla,
        Vs ~ (Vla - V0_la) + (Vlv - V0_lv) + Vas + Vvs + (Vra - V0_ra) + (Vrv - V0_rv) + Vap + Vvp
    ]
    @mtkbuild cv_odes = ODESystem(eqs, t)

    u0 = (Vlv=100.0, Vrv=100.0, Vap=40.0, Vra=10.0, Vvp=60.0, Vas=150.0, Vvs=270.0)

    return CV8Eed(cv_model=cv_odes, u0=u0, driving_pressure=driving_pressure)
end
