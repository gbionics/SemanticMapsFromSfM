import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from src.segment_3D_field import (
    assign_cluster_colors,
    save_clustered_ckpt,
    render_previews,
    make_viewer_render_fn,
    load_splats_from_ckpt,
    GsplatViewer,
)


def main(ckpt: str, clusters_pkl: str, out: str = None, previews: bool = True, viewer: bool = False):
    # load clustering results
    with open(clusters_pkl, 'rb') as f:
        data = pickle.load(f)

    # Expect 'labels' key
    if 'labels' in data:
        labels = data['labels']
    elif 'labels' in data:
        labels = data['labels']
    else:
        raise RuntimeError(f"clusters file {clusters_pkl} does not contain 'labels' key")

    labels = np.asarray(labels)

    # generate colors per gaussian
    colors = assign_cluster_colors(labels)

    # determine output path
    ckpt_p = Path(ckpt)
    if out is None:
        out_p = ckpt_p.with_name(ckpt_p.stem + "_clustered.pt")
    else:
        out_p = Path(out)
        if out_p.is_dir() or out_p.suffix != '.pt':
            out_p = out_p / (ckpt_p.stem + "_clustered.pt")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    # save clustered checkpoint (uses functions from segment_3D_field)
    save_clustered_ckpt(str(ckpt_p), str(out_p), colors)
    print(f"Clustered checkpoint saved to: {out_p}")

    # load splats for previews/viewer
    splats = load_splats_from_ckpt(str(out_p), device=("cuda" if torch.cuda.is_available() else "cpu"))

    preview_dir = out_p.parent / "cluster_previews"
    if previews:
        print("Rendering preview images...")
        render_previews(splats, str(preview_dir))
        print(f"Previews saved to: {preview_dir}")

    if viewer:
        try:
            import viser
            from nerfview import CameraState
            server = viser.ViserServer(port=8080, verbose=False)
            render_fn = make_viewer_render_fn(splats, device=("cuda" if torch.cuda.is_available() else "cpu"))
            viewer_app = GsplatViewer(server=server, render_fn=render_fn, output_dir=preview_dir, mode="rendering")
            print("Viewer running at http://localhost:8080 — press Ctrl+C to exit")
            try:
                while True:
                    pass
            except KeyboardInterrupt:
                print("Viewer stopped by user")
        except Exception as e:
            print("Interactive viewer unavailable:", e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('ckpt', help='Path to original 2DGS/3DGS checkpoint (.pt)')
    p.add_argument('clusters', help='Path to clustering results .pkl produced by gaussian_clustering')
    p.add_argument('--out', help='Output clustered checkpoint file or directory', default=None)
    p.add_argument('--no-previews', dest='previews', action='store_false', help='Disable preview rendering')
    p.add_argument('--viewer', dest='viewer', action='store_true', help='Launch interactive viewer after saving')
    args = p.parse_args()

    main(args.ckpt, args.clusters, out=args.out, previews=args.previews, viewer=args.viewer)
