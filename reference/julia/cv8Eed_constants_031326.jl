cv8Eed_ParAll = @with_kw (
    Emax_LV = 3.0, Blv = 0.033,
    Emax_RV = 0.7, Brv = 0.023, 
    Emax_LA = 0.48, Bla = 0.058,
    Emax_RA = 0.38, Bra = 0.046,
    τ = 25.0, Tmax = 200.0, 
    τ_a = 20.0, Tmax_a = 125.0, 
    HR = 75.0, AVD = 120.0,
    Eap = 0.26, Cvp = 8.0,
    Cas = 2.5, Cvs = 70.0, 
    Ras = 900.0, Rap = 23.0,  
    Vs = 850.0, 
    # Fixed parameters
    # Vascular
    Rcs = 20.0, Rcp = 10.0,
    Rra = 25.0, Rla = 15.0,
    # Valvular
    Rmv = 2.5, Rtv = 2.5,
    # Eed_ref: passive slope at Vref (replaces A in original parameterization)
    # Nominal values derived from A*B*exp(B*(Vref-V0)) with original A,B defaults
    Eedref_lv = 0.2,
    Eedref_rv = 0.08,
    Eedref_la = 0.46,
    Eedref_ra = 0.13,
    # Unstressed volumes
    V0_lv = 5.0, V0_rv = 5.0,
    V0_la = 5.0, V0_ra = 5.0,
    # Reference volumes (fixed anchors for Eed_ref parameterization)
    Vref_lv = 120.0, Vref_rv = 105.0,
    Vref_la = 60.0, Vref_ra = 45.0)

cv8Eed_ParAll_HIGH = @with_kw (
    Emax_LV = 5.0, Blv = 0.06, 
    Emax_RV = 4.0, Brv = 0.06, 
    Emax_LA = 1.0, Bla = 0.12,
    Emax_RA = 0.8, Bra = 0.1,
    τ = 80.0, Tmax = 400.0, 
    τ_a = 80.0, Tmax_a = 250.0, 
    HR = 133.0, AVD = 300.0,
    Eap = 5.0, Cvp = 9.0,
    Cas = 2.8, Cvs = 120.0, 
    Ras = 2200.0, Rap = 1200.0,  
    Vs = 3000.0, 
    # Fixed parameters
    Rcs = 20.0, Rcp = 10.0,
    Rra = 25.0, Rla = 15.0,
    Rmv = 2.5, Rtv = 2.5,
    # Eed_ref upper bounds
    Eedref_lv = 2.0,
    Eedref_rv = 1.5,
    Eedref_la = 2.5,
    Eedref_ra = 1.5,
    # Unstressed volumes
    V0_lv = 5.0, V0_rv = 5.0,
    V0_la = 5.0, V0_ra = 5.0,
    # Reference volumes
    Vref_lv = 120.0, Vref_rv = 105.0,
    Vref_la = 60.0, Vref_ra = 45.0)

cv8Eed_ParAll_LOW = @with_kw (
    Emax_LV = 0.5, Blv = 0.02, 
    Emax_RV = 0.4, Brv = 0.02, 
    Emax_LA = 0.1, Bla = 0.03,
    Emax_RA = 0.1, Bra = 0.03,
    τ = 15.0, Tmax = 100.0, 
    τ_a = 20.0, Tmax_a = 50.0, 
    HR = 43.0, AVD = 60.0,
    Eap = 0.03, Cvp = 1.0,
    Cas = 0.35, Cvs = 30.0, 
    Ras = 650.0, Rap = 10.0,  
    Vs = 300.0, 
    # Fixed parameters
    Rcs = 20.0, Rcp = 10.0,
    Rra = 25.0, Rla = 15.0,
    Rmv = 2.5, Rtv = 2.5,
    # Eed_ref lower bounds
    Eedref_lv = 0.08,
    Eedref_rv = 0.03,
    Eedref_la = 0.1,
    Eedref_ra = 0.03,
    # Unstressed volumes
    V0_lv = 5.0, V0_rv = 5.0,
    V0_la = 5.0, V0_ra = 5.0,
    # Reference volumes
    Vref_lv = 120.0, Vref_rv = 105.0,
    Vref_la = 60.0, Vref_ra = 45.0)


cv8Eed_ParFix = @with_kw (
    # Vascular
    Rcs = 20.0, Rcp = 10.0,
    Rra = 25.0, Rla = 15.0,
    # Valvular
    Rmv = 2.5, Rtv = 2.5,
    # Unstressed volumes
    V0_lv = 5.0, V0_rv = 5.0,
    V0_la = 5.0, V0_ra = 5.0,
    # Reference volumes (fixed anchors, not optimized)
    Vref_lv = 120.0, Vref_rv = 105.0,
    Vref_la = 60.0, Vref_ra = 45.0)

# Wave, summary, and cycle maps are identical to cv8
cv8Eed_rhc_wave_name_map = cv8_rhc_wave_name_map
cv8Eed_wave_name_map = cv8_wave_name_map
cv8Eed_summary_wave_map = cv8_summary_wave_map()
cv8Eed_cycle_map = cv8_cycle_map
