; Inspect the pixels of flight-2 plots 1001 and 1024 where the plot file is all-zero but the cube is not.
pro inspect_f2
  compile_opt idl2
  B = 'D:\Biomass and Height data for modeling(Ibrahim)\Biomass Experiment\2025_NUE_AGT_I_DPIRD\2025-11-13\flight2'
  e = ENVI(/HEADLESS)
  cube = e.OpenRaster(B + '\20251113_AGT_NUE_Bejoording_Bejoording_Gobi_flight2_VNIR_Orthomosaic.bin')
  csr = cube.SPATIALREF & ctp = double(csr.TIE_POINT_MAP) & ps = double((csr.PIXEL_SIZE)[0]) & nb = cube.NBANDS
  s = obj_new('IDLffShape', B + '\aligned_grid_refit_flight2.shp')
  s.GetProperty, N_ENTITIES=np, ATTRIBUTE_NAMES=an
  ipid = (where(strupcase(an) eq 'PLOT_ID'))[0]
  foreach pid, [1001L, 1024L] do begin
    for i = 0, np-1 do begin
      ent = s.GetEntity(i, /ATTRIBUTES)
      if (*ent.attributes).(ipid) eq pid then break
      s.DestroyEntity, ent
    endfor
    v = *ent.vertices & s.DestroyEntity, ent
    poly = rv3_inset(v, 0d)
    f = (file_search(B + '\dataset\vnir\plot_' + strtrim(pid, 2) + '_*.tif'))[0]
    r = e.OpenRaster(f) & sr = r.SPATIALREF & tp = double(sr.TIE_POINT_MAP)
    nc = r.NCOLUMNS & nr = r.NROWS
    col0 = round((tp[0] - ctp[0]) / ps) & row0 = round((ctp[1] - tp[1]) / ps)
    tif = r.GetData(INTERLEAVE='bsq') & r.Close
    sub = cube.GetData(INTERLEAVE='bsq', SUB_RECT=[col0, row0, col0+nc-1, row0+nr-1])
    cols = rebin(dindgen(nc), nc, nr) + col0 & rows = rebin(transpose(dindgen(nr)), nc, nr) + row0
    xc = ctp[0] + (cols + 0.5d) * ps & yc = ctp[1] - (rows + 0.5d) * ps
    m = reform(rv3_inside(poly, xc[*], yc[*]), nc, nr)
    tif_nz = total(tif ne 0, 3) gt 0 & cube_nz = total(sub ne 0, 3) gt 0
    print, 'plot ', pid, ': window ', nc, 'x', nr, ' polygon px (my raster) ', long(total(m)), ' tif nonzero ', long(total(tif_nz)), $
           ' tif nonzero & in polygon ', long(total(tif_nz and m)), ' tif nonzero outside polygon ', long(total(tif_nz and (m eq 0)))
    print, '   cube NaN count in window: ', long(total(~finite(sub))), '  negative values: ', long(total(sub lt 0)), $
           '  tif NaN: ', long(total(~finite(tif))), '  tif negative: ', long(total(tif lt 0))
    w = where(m and (tif_nz eq 0), nw)
    print, '   polygon pixels all-zero in tif: ', nw
    for k = 0, (nw < 8) - 1 do begin
      c = w[k] mod nc & rr = w[k] / nc
      cv = reform(sub[c, rr, *]) & tv = reform(tif[c, rr, *])
      print, '   px col/row ', c, rr, ' cube: nonzero bands ', long(total(cv ne 0)), ' min ', min(cv), ' max ', max(cv), ' NaN ', long(total(~finite(cv))), $
             ' | tif: nonzero bands ', long(total(tv ne 0)), ' NaN ', long(total(~finite(tv)))
      print, '      cube bands 1-6: ', cv[0:5], '  band 78/115: ', cv[77], cv[114]
    endfor
    w2 = where((m eq 0) and tif_nz, nw2)
    print, '   tif nonzero pixels outside my polygon raster: ', nw2
  endforeach
  obj_destroy, s & cube.Close & e.Close
end
