# Animations: from a land-focused map to a coastline-focused map

Draft animations that start from the default pyseas world map (Equal Earth, centred on 0°) and transform it into the Elastic-X projection (`projection/elastic-X.h5`). The red lines show where Elastic-X cuts the map.

There are two variants, and each comes in two colour styles:

- `poc_warp/`: the flat map pans to 72.5°E, then tears along the cut lines and warps into Elastic-X.
- `poc_globe/`: the flat map wraps onto a globe, rotates, then unpeels into Elastic-X.

| Style | Colours | Flat warp | Via the globe |
|---|---|---|---|
| `bathy` (default) | GFW-dark bathymetry; opens on the flat pyseas map and cross-fades to the texture | `poc_warp/warp_1080p.mp4` | `poc_globe/globe_1080p.mp4` |
| `topo` | cmocean `topo` from -5000 to 5000 m; topography shows from the first frame | `poc_warp/warp_1080p_topo.mp4` | `poc_globe/globe_1080p_topo.mp4` |

The contact sheets and `*_key_*.png` files are stills from these videos. The `poc_*` files are the original low-resolution proofs of concept.

## Data (not in git)

The scripts expect these files in `common/`:

- `etopo1_0p1deg.tif`: ETOPO1 bedrock at 0.1°, used for the bathy style.
- `etopo1_ice_0p05deg.nc`: ETOPO1 Ice Surface at 0.05°, used for the topo style. Download it from NOAA ERDDAP:
  `curl -g "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude[(-90):3:(90)][(-180):3:(180)]" -o etopo1_ice_0p05deg.nc`

Caches, textures and frame folders are git-ignored because the scripts regenerate them.

## Re-running

You need Python 3.9 with pyseas, cartopy, matplotlib, h5py, scipy, rasterio, cmocean, opencv and imageio-ffmpeg.

```
cd poc_warp
python make_texture_hd.py && python precompute_hd.py && python render_hd.py [--style topo]

cd poc_globe
python make_texture_hd.py && python precompute_hd.py && python precompute_topo.py
python render_hd.py [--style topo]
python make_video_hd.py --frames frames_1080p_topo --name globe_1080p_topo --key-times 0,0.5,6.1,9.8,end   # topo
```

## Known issues

These are drafts, not production code.

- Flat warp: the Antarctic strip folds over itself in a translucent sheet. A black wedge appears at the Drake Passage neck. A translucent copy of the southeast Pacific shows briefly as the warp starts.
- Both variants: ETOPO1 survey-track streaks show in the deep ocean, and relief smears along the outer rim of the Elastic-X map.
