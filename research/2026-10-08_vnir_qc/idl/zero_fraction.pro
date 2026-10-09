; Per plot and per band: fraction of polygon pixels that are exactly 0 in the ORIGINAL cube.
pro zero_fraction
  compile_opt idl2
  plot_dir  = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\Biomass Dataset\NUE_I_DPIRD_Muresk_F_Gobi\2025-09-30\vnir'
  cube_file = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\20250930_NUE_I_DPIRD_Muresk_F_Gobi_VNIR_Orthomosaic.bin'
  out_dir   = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025-09-30\vnir_qc_2026-10-08'
  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = csr.TIE_POINT_MAP & cps = csr.PIXEL_SIZE
  nb = cube.NBANDS & wl = cube.METADATA['wavelength']
  files = file_search(plot_dir + '\plot_*.tif', COUNT=nf)
  ids = lonarr(nf) & names = strarr(nf)
  for i = 0, nf-1 do begin
    s = stregex(file_basename(files[i]), 'plot_([0-9]+)_([A-Za-z0-9]+)\.tif', /SUBEXPR, /EXTRACT)
    ids[i] = long(s[1]) & names[i] = s[2]
  endfor
  o = sort(ids) & files = files[o] & ids = ids[o] & names = names[o]
  zf = dblarr(nb, nf)
  openw, lun, out_dir + '\plot_zero_fraction_inside_polygon_from_cube.csv', /GET_LUN
  printf, lun, 'Plot_ID,name,n_box_pixels,n_polygon_pixels,frac_box_outside_polygon,frac_polygon_px_with_any_zero_band,' + $
               'frac_zero_420nm,frac_zero_441nm,frac_zero_470nm,frac_zero_550nm,frac_zero_670nm,frac_zero_800nm'
  for i = 0, nf-1 do begin
    r = e.OpenRaster(files[i])
    sr = r.SPATIALREF & tp = sr.TIE_POINT_MAP
    nc = r.NCOLUMNS & nr = r.NROWS
    col0 = round((tp[0] - ctp[0]) / cps[0]) & row0 = round((ctp[1] - tp[1]) / cps[1])
    tif = r.GetData(INTERLEAVE='bsq')
    r.Close
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[col0, row0, col0+nc-1, row0+nr-1])
    m = total(tif ne 0, 3) gt 0
    win = where(m, n_in)
    anyzero = total(sub eq 0, 3) gt 0
    for b = 0, nb-1 do begin
      band = sub[*, *, b]
      zf[b, i] = total(band[win] eq 0) / double(n_in)
    endfor
    f = '(F0.4)'
    printf, lun, strjoin([strtrim(ids[i],2), names[i], strtrim(long(nc)*nr,2), strtrim(n_in,2), string(1d - double(n_in)/(long(nc)*nr), format=f), $
      string(total(anyzero[win]) / double(n_in), format=f), string(zf[5,i], format=f), string(zf[12,i], format=f), string(zf[20,i], format=f), $
      string(zf[43,i], format=f), string(zf[77,i], format=f), string(zf[114,i], format=f)], ',')
  endfor
  free_lun, lun
  openw, lun2, out_dir + '\band_zero_fraction_inside_polygon_from_cube.csv', /GET_LUN
  printf, lun2, 'band,wavelength_nm,mean_zero_frac_over_plots,min_zero_frac,max_zero_frac'
  for b = 0, nb-1 do printf, lun2, strtrim(b+1,2), ',', string(wl[b], format='(F0.1)'), ',', string(mean(zf[b,*]), format='(F0.4)'), ',', $
      string(min(zf[b,*]), format='(F0.4)'), ',', string(max(zf[b,*]), format='(F0.4)')
  free_lun, lun2
  cube.Close & e.Close
  print, 'zero fraction done'
end
