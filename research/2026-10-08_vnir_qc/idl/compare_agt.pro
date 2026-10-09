; AGT anthesis (2025-09-22): every per-plot VNIR GeoTIFF vs the same window of the original ENVI cube,
; all bands, plus per-band zero fractions inside the polygon (from the cube). Headless ENVI 6.2 / IDL 9.2.
pro compare_agt
  compile_opt idl2
  plot_dir  = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-09-22\dataset\vnir'
  cube_file = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-09-22\20250922_AGT_NUE_Bejoording_Bejoording_Gobi_VNIR_Orthomosaic.bin'
  out_dir   = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-09-22\vnir_qc_2026-10-08'
  file_mkdir, out_dir
  t0 = systime(1)
  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(cube_file)
  csr = cube.SPATIALREF & ctp = csr.TIE_POINT_MAP & cps = csr.PIXEL_SIZE
  nb = cube.NBANDS & wl = cube.METADATA['wavelength']
  print, 'cube: ', cube.NCOLUMNS, cube.NROWS, nb, ' ', cube.DATA_TYPE, ' crs ', csr.COORD_SYS_CODE, ' ps ', cps
  idx = read_csv(plot_dir + '\plots_vnir_index.csv', HEADER=hdr)
  idx_id = long(idx.FIELD1) & idx_px = long(idx.FIELD6)
  files = file_search(plot_dir + '\plot_*.tif', COUNT=nf)
  ids = lonarr(nf) & names = strarr(nf)
  for i = 0, nf-1 do begin
    s = stregex(file_basename(files[i]), 'plot_([0-9]+)_([A-Za-z0-9]+)\.tif', /SUBEXPR, /EXTRACT)
    ids[i] = long(s[1]) & names[i] = s[2]
  endfor
  o = sort(ids) & files = files[o] & ids = ids[o] & names = names[o]
  print, 'plots found: ', nf
  openw, lun, out_dir + '\plot_vs_cube_comparison.csv', /GET_LUN
  printf, lun, 'Plot_ID,name,tif_cols,tif_rows,tif_bands,tif_dtype,tif_crs,tif_pixel_m,cube_col0,cube_row0,align_resid_x_m,align_resid_y_m,' + $
               'n_inside_tif,n_inside_index,n_cube_allzero_in_box,n_outside_poly_with_cube_data,n_inside_px_any_band_diff,max_abs_diff_inside,' + $
               'frac_inside_bitexact,ndvi_tif,ndvi_cube,frac_polygon_px_any_zero_band,frac_zero_417nm,frac_zero_441nm,frac_zero_470nm,frac_zero_550nm,frac_zero_670nm,frac_zero_800nm'
  band_maxdiff = lonarr(nb) & zsum = dblarr(nb) & zmax = dblarr(nb) & zmin = dblarr(nb) + 1 & nz = 0L
  for i = 0, nf-1 do begin
    r = e.OpenRaster(files[i])
    sr = r.SPATIALREF
    nc = r.NCOLUMNS & nr = r.NROWS & nbt = r.NBANDS & dtype = r.DATA_TYPE
    tp = sr.TIE_POINT_MAP & ps = sr.PIXEL_SIZE & crs = sr.COORD_SYS_CODE
    fx = (tp[0] - ctp[0]) / cps[0] & fy = (ctp[1] - tp[1]) / cps[1]
    col0 = round(fx) & row0 = round(fy)
    resx = (fx - col0) * cps[0] & resy = (fy - row0) * cps[1]
    if (col0 lt 0) or (row0 lt 0) or (col0+nc gt cube.NCOLUMNS) or (row0+nr gt cube.NROWS) then begin
      print, 'plot ', ids[i], ' window outside cube' & r.Close & continue
    endif
    tif = r.GetData(INTERLEAVE='bsq')
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[col0, row0, col0+nc-1, row0+nr-1])
    r.Close
    m = total(tif ne 0, 3) gt 0
    n_in = long(total(m))
    if n_in eq 0 then begin
      print, 'plot ', ids[i], ' has no non-zero pixels' & continue
    endif
    cube_nz = total(sub ne 0, 3) gt 0
    n_cube_zero = long(total(cube_nz eq 0))
    n_out_data = long(total((m eq 0) and cube_nz))
    diff = long(tif) - long(sub)
    dmask = total(diff ne 0, 3) gt 0
    n_diff_in = long(total(dmask and m))
    m3 = rebin(byte(m), nc, nr, nb)
    ad = abs(diff) * m3
    maxd = max(ad)
    for b = 0, nb-1 do band_maxdiff[b] = band_maxdiff[b] > max(ad[*, *, b])
    win = where(m)
    nir_t = float(tif[*, *, 114]) & red_t = float(tif[*, *, 77])
    nir_c = float(sub[*, *, 114]) & red_c = float(sub[*, *, 77])
    ndvi_t = mean(((nir_t - red_t) / ((nir_t + red_t) > 1))[win])
    ndvi_c = mean(((nir_c - red_c) / ((nir_c + red_c) > 1))[win])
    anyzero = total(sub eq 0, 3) gt 0
    zf = dblarr(nb)
    for b = 0, nb-1 do begin
      band = sub[*, *, b] & zf[b] = total(band[win] eq 0) / double(n_in)
    endfor
    zsum += zf & zmax = zmax > zf & zmin = zmin < zf & nz++
    j = (where(idx_id eq ids[i], cnt))[0]
    n_idx = cnt gt 0 ? idx_px[j] : -1L
    f4 = '(F0.4)'
    printf, lun, strjoin([strtrim(ids[i],2), names[i], strtrim(nc,2), strtrim(nr,2), strtrim(nbt,2), dtype, strtrim(crs,2), string(ps[0], format='(F0.3)'), $
      strtrim(col0,2), strtrim(row0,2), string(resx, format='(F0.5)'), string(resy, format='(F0.5)'), strtrim(n_in,2), strtrim(n_idx,2), strtrim(n_cube_zero,2), $
      strtrim(n_out_data,2), strtrim(n_diff_in,2), strtrim(maxd,2), string(1d - double(n_diff_in)/n_in, format='(F0.6)'), $
      string(ndvi_t, format=f4), string(ndvi_c, format=f4), string(total(anyzero[win]) / double(n_in), format=f4), $
      string(zf[5], format=f4), string(zf[12], format=f4), string(zf[20], format=f4), string(zf[43], format=f4), string(zf[77], format=f4), string(zf[114], format=f4)], ',')
    if (i mod 100) eq 0 then print, 'plot ', ids[i], ' inside ', n_in, ' diff px ', n_diff_in, ' maxdiff ', maxd, ' t=', long(systime(1)-t0), 's'
  endfor
  free_lun, lun
  openw, lun3, out_dir + '\band_max_abs_diff_and_zero_fraction.csv', /GET_LUN
  printf, lun3, 'band,wavelength_nm,max_abs_diff_inside_all_plots,mean_zero_frac_over_plots,min_zero_frac,max_zero_frac'
  for b = 0, nb-1 do printf, lun3, strtrim(b+1,2), ',', string(wl[b], format='(F0.2)'), ',', strtrim(band_maxdiff[b],2), ',', $
      string(zsum[b] / nz, format='(F0.4)'), ',', string(zmin[b], format='(F0.4)'), ',', string(zmax[b], format='(F0.4)')
  free_lun, lun3
  cube.Close & e.Close
  print, 'done ', nz, ' plots in ', long(systime(1)-t0), ' s'
end
