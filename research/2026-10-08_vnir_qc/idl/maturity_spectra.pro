; Mean spectrum (band > 0 only) of the first N plots (polygon inset 10 cm) for the maturity cubes.
pro ms_one, e, cube_file, shp_file, label, lun, nplots
  compile_opt idl2
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = double(csr.TIE_POINT_MAP) & ps = double((csr.PIXEL_SIZE)[0])
  nb = cube.NBANDS & wl = double(cube.METADATA['wavelength'])
  print, label, ': ', cube.NCOLUMNS, cube.NROWS, nb, ' ps ', ps, ' crs ', csr.COORD_SYS_CODE, ' wl ', wl[0], wl[-1]
  s = obj_new('IDLffShape', shp_file)
  s.GetProperty, N_ENTITIES=np, ATTRIBUTE_NAMES=an
  ipid = (where(strupcase(an) eq 'PLOT_ID'))[0]
  b670 = (where(abs(wl - 670) eq min(abs(wl - 670))))[0] & b800 = (where(abs(wl - 800) eq min(abs(wl - 800))))[0]
  step = (np / nplots) > 1
  for i = 0, np - 1, step do begin
    ent = s.GetEntity(i, /ATTRIBUTES)
    v = *ent.vertices & att = *ent.attributes & pid = att.(ipid)
    s.DestroyEntity, ent
    poly = rv3_inset(v, 0.10d)
    c0 = floor((min(poly[0, *]) - ctp[0]) / ps) & c1 = ceil((max(poly[0, *]) - ctp[0]) / ps)
    r0 = floor((ctp[1] - max(poly[1, *])) / ps) & r1 = ceil((ctp[1] - min(poly[1, *])) / ps)
    if (c0 lt 0) or (r0 lt 0) or (c1 ge cube.NCOLUMNS) or (r1 ge cube.NROWS) then begin
      print, label, ' plot ', pid, ' outside cube' & continue
    endif
    nc = c1 - c0 + 1 & nr = r1 - r0 + 1
    cols = rebin(dindgen(nc), nc, nr) + c0 & rows = rebin(transpose(dindgen(nr)), nc, nr) + r0
    xc = ctp[0] + (cols + 0.5d) * ps & yc = ctp[1] - (rows + 0.5d) * ps
    m = rv3_inside(poly, xc[*], yc[*]) & idx = where(m, n_all)
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[c0, r0, c1, r1])
    spx = (reform(sub, long(nc) * nr, nb))[idx, *]
    spec = dblarr(nb) + !values.d_nan
    for b = 0, nb-1 do begin
      col = double(spx[*, b]) & w = where(col gt 0, nw)
      if nw gt 0 then spec[b] = mean(col[w])
    endfor
    r670 = double(spx[*, b670]) & r800 = double(spx[*, b800]) & ok = (r670 gt 0) and (r800 gt 0)
    w = where(ok, nw)
    ndvi = nw gt 0 ? mean((r800[w] - r670[w]) / (r800[w] + r670[w])) : !values.d_nan
    allzero = total(total(spx eq 0, 2) eq nb) / n_all
    vmin = min(spx, max=vmax)
    wnz = where(spx gt 0, nnz)
    nzmin = nnz gt 0 ? min(spx[wnz]) : !values.d_nan
    frac_lt1 = nnz gt 0 ? total(spx[wnz] lt 1) / nnz : !values.d_nan
    printf, lun, strjoin([label, strtrim(pid, 2), strtrim(n_all, 2), string(allzero, format='(F0.3)'), string(vmin, format='(G0.5)'), $
                          string(nzmin, format='(G0.5)'), string(vmax, format='(G0.5)'), string(frac_lt1, format='(F0.3)'), string(ndvi, format='(F0.4)'), $
                          strtrim(string(spec, format='(G0.5)'), 2)], ',')
    print, label, ' plot ', pid, ' px ', n_all, ' all-zero ', allzero, ' min/nzmin/max ', vmin, nzmin, vmax, ' frac(0<v<1) ', frac_lt1, $
           ' NDVI ', ndvi, ' R670 ', spec[b670], ' R800 ', spec[b800]
  endfor
  obj_destroy, s & cube.Close
end

pro maturity_spectra
  compile_opt idl2
  out_dir = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\vnir_qc_2026-10-08'
  e = ENVI(/HEADLESS)
  openw, lun, out_dir + '\maturity_cubes_mean_spectra_first_plots.csv', /GET_LUN
  printf, lun, 'flight,Plot_ID,region_px,frac_px_all_bands_zero,min_value,min_nonzero_value,max_value,frac_nonzero_below_1,NDVI_800_670,mean_spectrum_172_bands_raw_units'
  ms_one, e, 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-11-13\flight1\20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight1_VNIR_Orthomosaic.bin', $
          'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-11-13\flight1\aligned_grid_refit_flight1.shp', 'agt_2025-11-13_f1', lun, 12
  ms_one, e, 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-11-13\flight2\20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin', $
          'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-11-13\flight2\aligned_grid_refit_flight2.shp', 'agt_2025-11-13_f2', lun, 8
  ms_one, e, 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-09-22\20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin', $
          'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-09-22\grid_qa\refit_auto3.shp', 'agt_2025-09-22', lun, 4
  ms_one, e, 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-11-21\20251121_NUE_I_DPIRD_Muresk_F_Gobi_noextent_VNIR_Orthomosaic.bin', $
          'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Grid_Refit_Comparison\aligned_grid_refit.shp', 'muresk_2025-11-21', lun, 4
  free_lun, lun
  e.Close
  print, 'maturity spectra done'
end
