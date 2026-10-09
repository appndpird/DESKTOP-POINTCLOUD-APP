; Per flight: for every plot (sampling region = polygon inset 10 cm, pixel-centre rasterisation) and every band,
; the fraction of region pixels with a valid (non-zero) value in the ORIGINAL cube. Writes
;   <out_dir>\band_valid_fraction_<label>.csv   Plot_ID,name,region_px,frac_in_cube,<172 wavelength columns>
; Needs rv3_inset / rv3_inside from replicate_vnir_v3.pro.
pro band_qc_flight, e, cube_file, shp_file, out_csv, label
  compile_opt idl2
  t0 = systime(1)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = double(csr.TIE_POINT_MAP) & ps = double((csr.PIXEL_SIZE)[0])
  nb = cube.NBANDS & wl = double(cube.METADATA['wavelength'])
  s = obj_new('IDLffShape', shp_file)
  s.GetProperty, N_ENTITIES=np, ATTRIBUTE_NAMES=an
  ipid = (where(strupcase(an) eq 'PLOT_ID'))[0]
  inm = (where(strupcase(an) eq 'B/R'))[0]
  openw, lun, out_csv, /GET_LUN
  printf, lun, 'Plot_ID,name,region_px,frac_in_cube,' + strjoin(strtrim(string(wl, format='(F0.1)'), 2), ',')
  for i = 0, np-1 do begin
    ent = s.GetEntity(i, /ATTRIBUTES)
    v = *ent.vertices & att = *ent.attributes
    pid = att.(ipid) & name = inm ge 0 ? strtrim(att.(inm), 2) : ''
    s.DestroyEntity, ent
    poly = rv3_inset(v, 0.10d)
    c0 = floor((min(poly[0, *]) - ctp[0]) / ps) & c1 = ceil((max(poly[0, *]) - ctp[0]) / ps)
    r0 = floor((ctp[1] - max(poly[1, *])) / ps) & r1 = ceil((ctp[1] - min(poly[1, *])) / ps)
    nc = c1 - c0 + 1 & nr = r1 - r0 + 1
    cols = rebin(dindgen(nc), nc, nr) + c0 & rows = rebin(transpose(dindgen(nr)), nc, nr) + r0
    xc = ctp[0] + (cols + 0.5d) * ps & yc = ctp[1] - (rows + 0.5d) * ps
    m = reform(rv3_inside(poly, xc[*], yc[*]), nc, nr)
    n_region = long(total(m))
    inside_cube = (cols ge 0) and (rows ge 0) and (cols lt cube.NCOLUMNS) and (rows lt cube.NROWS)
    m_in = m and inside_cube
    n_in = long(total(m_in))
    vf = dblarr(nb)
    if n_in gt 0 then begin
      cc0 = c0 > 0 & rr0 = r0 > 0 & cc1 = c1 < (cube.NCOLUMNS - 1) & rr1 = r1 < (cube.NROWS - 1)
      sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[cc0, rr0, cc1, rr1])
      msub = m_in[cc0 - c0:cc1 - c0, rr0 - r0:rr1 - r0]
      idx = where(msub)
      spx = (reform(sub, long(cc1 - cc0 + 1) * (rr1 - rr0 + 1), nb))[idx, *]
      for b = 0, nb-1 do vf[b] = total(spx[*, b] gt 0) / double(n_region)
    endif
    printf, lun, strjoin([strtrim(pid, 2), name, strtrim(n_region, 2), string(double(n_in) / (n_region > 1), format='(F0.4)'), $
                          strtrim(string(vf, format='(F0.4)'), 2)], ',')
    if (i mod 200) eq 0 then print, label, ' plot ', pid, ' region ', n_region, ' in cube ', n_in, ' t=', long(systime(1) - t0), 's'
  endfor
  free_lun, lun
  obj_destroy, s & cube.Close
  print, label, ' done: ', np, ' plots, ', long(systime(1) - t0), ' s -> ', out_csv
end

pro band_qc
  compile_opt idl2
  B = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment'
  A = B + '\2025_NUE_AGT_I_DPIRD'
  e = ENVI(/HEADLESS)
  band_qc_flight, e, B + '\2025-09-30\20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin', B + '\2025-09-30\aligned_grid_refit_20250930.shp', $
                  B + '\2025-09-30\vnir_qc_2026-10-08\band_valid_fraction_muresk_2025-09-30.csv', 'muresk_2025-09-30'
  band_qc_flight, e, B + '\2025-11-21\20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin', B + '\Grid_Refit_Comparison\aligned_grid_refit.shp', $
                  B + '\2025-11-21\vnir_qc_2026-10-08\band_valid_fraction_muresk_2025-11-21.csv', 'muresk_2025-11-21'
  band_qc_flight, e, A + '\2025-09-22\20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin', A + '\2025-09-22\grid_qa\refit_auto3.shp', $
                  A + '\2025-09-22\vnir_qc_2026-10-08\band_valid_fraction_agt_2025-09-22.csv', 'agt_2025-09-22'
  band_qc_flight, e, A + '\2025-11-13\flight1\20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin', A + '\2025-11-13\flight1\aligned_grid_refit_flight1.shp', $
                  A + '\2025-11-13\flight1\vnir_qc_2026-10-08\band_valid_fraction_agt_2025-11-13_f1.csv', 'agt_2025-11-13_f1'
  band_qc_flight, e, A + '\2025-11-13\flight2\20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin', A + '\2025-11-13\flight2\aligned_grid_refit_flight2.shp', $
                  A + '\2025-11-13\flight2\vnir_qc_2026-10-08\band_valid_fraction_agt_2025-11-13_f2.csv', 'agt_2025-11-13_f2'
  e.Close
  print, 'band qc all done'
end
