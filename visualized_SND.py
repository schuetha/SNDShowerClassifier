"""
3D Event Display for SND@LHC GNN Data - X, Y, Z + Energy heatmap
=================================================================
Log scale by default so SciFi (small deposits) and MuFilter (large
deposits) are both visible on the same heatmap.

Flavour: 0=Background, 12=CC nue, 14=CC numu, 23=NC
Markers: circle=SciFi, square=MuFilter US, diamond=MuFilter DS

Usage:
    python plot_events_3d_xyz.py -i GNN_data.pt -o ./plots -n 5
    python plot_events_3d_xyz.py -i GNN_data.pt --linear
"""

import os, argparse
import numpy as np
import torch
import plotly.graph_objects as go

FEATURE_COLS = ['XY', 'Z', 'Energy', 'dettype', 'vertical']
COL = {name: idx for idx, name in enumerate(FEATURE_COLS)}

DETTYPE_CONFIG = {
    2: {'label': 'MuFilter US', 'symbol': 'square'},
    3: {'label': 'MuFilter DS', 'symbol': 'diamond'},
    4: {'label': 'SciFi',       'symbol': 'circle'},
}

FLAVOUR_LABELS = {0: 'Background', 12: 'CC nue', 14: 'CC numu', 23: 'NC'}


def load_data(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload['features'], payload['flavours']


def build_event_figure(hits, event_idx, flavour, use_log=True):
    fig = go.Figure()
    energy_all = hits[:, COL['Energy']]
    emin_raw = float(energy_all.min())
    emax_raw = float(energy_all.max())

    if use_log:
        safe_min = max(emin_raw, 1e-3)
        cmin = float(np.log10(safe_min))
        cmax = float(np.log10(max(emax_raw, safe_min * 10)))
        energy_label = 'log10(Energy) [a.u.]'
    else:
        cmin, cmax = emin_raw, emax_raw
        if cmin == cmax:
            cmax = cmin + 1.0
        energy_label = 'Energy [a.u.]'

    colorbar_shown = False
    for det_id, cfg in DETTYPE_CONFIG.items():
        mask = hits[:, COL['dettype']] == det_id
        sub = hits[mask]
        if len(sub) == 0:
            continue

        n_det = len(sub)
        legend_name = cfg['label'] + ' (' + str(n_det) + ')'

        xy   = sub[:, COL['XY']]
        z    = sub[:, COL['Z']]
        en   = sub[:, COL['Energy']]
        vert = sub[:, COL['vertical']]

        x_vals = np.where(vert == 1, xy, 0.0)
        y_vals = np.where(vert == 0, xy, 0.0)

        if use_log:
            c_vals = np.log10(np.clip(en, max(emin_raw, 1e-3), None))
        else:
            c_vals = en.copy()

        show_cbar = not colorbar_shown
        colorbar_shown = True

        hover_texts = []
        for x, y, zi, ei in zip(x_vals, y_vals, z, en):
            hover_texts.append(
                '<b>' + cfg['label'] + '</b><br>'
                'X: ' + f'{x:.2f}' + ' cm<br>'
                'Y: ' + f'{y:.2f}' + ' cm<br>'
                'Z: ' + f'{zi:.2f}' + ' cm<br>'
                'Energy: ' + f'{ei:.2f}' + ' a.u.'
            )

        cbar_dict = None
        if show_cbar:
            cbar_dict = dict(
                title=dict(text=energy_label, font=dict(size=13)),
                len=0.65, thickness=18, x=1.02,
                tickfont=dict(size=11),
            )

        fig.add_trace(go.Scatter3d(
            x=x_vals.tolist(), y=y_vals.tolist(), z=z.tolist(),
            mode='markers', name=legend_name,
            marker=dict(
                color=c_vals.tolist(), colorscale='Turbo',
                cmin=cmin, cmax=cmax, size=3.5,
                symbol=cfg['symbol'], opacity=0.88,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
                showscale=show_cbar, colorbar=cbar_dict,
            ),
            hovertext=hover_texts, hoverinfo='text',
        ))

    flav_name = FLAVOUR_LABELS.get(flavour, 'flavour ' + str(flavour))
    n_hits = len(hits)
    scale_tag = 'log' if use_log else 'linear'

    # ── Per-detector hit summary for annotation box ──
    det_lines = []
    for det_id in sorted(DETTYPE_CONFIG.keys()):
        cfg = DETTYPE_CONFIG[det_id]
        mask = hits[:, COL['dettype']] == det_id
        n = int(mask.sum())
        if n > 0:
            sub_en = hits[mask][:, COL['Energy']]
            det_lines.append(
                cfg['label'].ljust(12) + ':  '
                + str(n).rjust(4) + ' hits   '
                + 'E: ' + f'{float(sub_en.min()):.1f}'
                + ' - ' + f'{float(sub_en.max()):.1f}'
            )
        else:
            det_lines.append(
                cfg['label'].ljust(12) + ':     0 hits'
            )

    summary_text = (
        '<b>Hit Summary</b><br>'
        + '<br>'.join(det_lines)
        + '<br>\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500'
          '\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500<br>'
        + '<b>Total' + ' ' * 9 + ':  '
        + str(n_hits).rjust(4) + ' hits</b>'
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X [cm]', gridcolor='rgba(255,255,255,0.06)',
                       backgroundcolor='#13151f', showbackground=True),
            yaxis=dict(title='Y [cm]', gridcolor='rgba(255,255,255,0.06)',
                       backgroundcolor='#151727', showbackground=True),
            zaxis=dict(title='Z [cm] (beam)', gridcolor='rgba(255,255,255,0.06)',
                       backgroundcolor='#14162a', showbackground=True),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        title=dict(
            text='Event ' + str(event_idx) + ' | ' + flav_name + ' | '
                 + str(n_hits) + ' hits | Energy: '
                 + f'{emin_raw:.1f}' + '-' + f'{emax_raw:.1f}'
                 + ' a.u. (' + scale_tag + ')',
            font=dict(size=15, color='#eee'), x=0.5,
        ),
        annotations=[
            dict(
                text=summary_text,
                align='left',
                showarrow=False,
                xref='paper', yref='paper',
                x=0.01, y=0.28,
                bordercolor='rgba(124,138,255,0.3)',
                borderwidth=1, borderpad=10,
                bgcolor='rgba(20,22,40,0.90)',
                font=dict(family='monospace', size=11, color='#ccc'),
            ),
        ],
        paper_bgcolor='#0f1117', plot_bgcolor='#0f1117',
        showlegend=True,
        legend=dict(x=0.01, y=0.98, bgcolor='rgba(20,22,35,0.85)',
                    bordercolor='rgba(255,255,255,0.1)', borderwidth=1,
                    font=dict(color='#ccc', size=11)),
        margin=dict(l=0, r=0, t=50, b=0),
        width=1400, height=900,
    )
    return fig


def save_all(features, flavours, out_dir, max_events, use_log):
    unique_flavours = sorted(set(flavours.numpy().tolist()))

    for flav in unique_flavours:
        indices = [i for i, f in enumerate(flavours.tolist()) if f == flav][:max_events]
        if not indices:
            continue

        flav_name = FLAVOUR_LABELS.get(flav, 'flavour_' + str(flav))
        safe_name = flav_name.replace(' ', '_')
        flav_dir = os.path.join(out_dir, safe_name)
        os.makedirs(flav_dir, exist_ok=True)

        event_files = []
        for idx in indices:
            hits = features[idx].numpy().astype(np.float32)
            if len(hits) == 0:
                continue
            fig = build_event_figure(hits, idx, flav, use_log=use_log)
            fname = 'event_' + str(idx) + '.html'
            fig.write_html(os.path.join(flav_dir, fname), include_plotlyjs='cdn')
            event_files.append((idx, fname))

            for det_id, cfg in DETTYPE_CONFIG.items():
                det_mask = hits[:, COL['dettype']] == det_id
                sub_en = hits[det_mask][:, COL['Energy']]
                if len(sub_en) > 0:
                    print('    ' + cfg['label'].ljust(15) + ': '
                          + str(len(sub_en)).rjust(4) + ' hits, E = '
                          + f'{sub_en.min():.2f}' + ' - '
                          + f'{sub_en.max():.2f}' + ' a.u.')
            print('  -> ' + safe_name + '/' + fname + '  (' + str(len(hits)) + ' hits)\n')

        if not event_files:
            continue

        # Build index.html with navigation
        _build_index(flav_dir, flav_name, event_files, use_log)
        print('  Index -> ' + safe_name + '/index.html  (' + str(len(event_files)) + ' events)\n')


def _build_index(flav_dir, flav_name, event_files, use_log):
    scale_label = 'log' if use_log else 'linear'
    nav = ''
    for eidx, fn in event_files:
        nav += ('<button onclick="loadEvent(\'' + fn + '\')" '
                'id="btn-' + str(eidx) + '">Event ' + str(eidx) + '</button>')

    html = ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<title>' + flav_name + ' - SND@LHC 3D Event Display</title>\n'
            '<style>\n'
            '*{margin:0;padding:0;box-sizing:border-box}\n'
            "body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0}\n"
            'header{background:linear-gradient(135deg,#1a1d2e,#2a1f3d);padding:16px 28px;'
            'border-bottom:1px solid rgba(255,255,255,0.08);display:flex;align-items:center;'
            'gap:16px;flex-wrap:wrap}\n'
            'header h1{font-size:1.3rem;font-weight:600}\n'
            'header h1 span{color:#7c8aff;font-weight:700}\n'
            '.badge{padding:4px 12px;border-radius:16px;font-size:0.8rem;'
            'background:rgba(124,138,255,0.15);border:1px solid rgba(124,138,255,0.3);color:#a8b4ff}\n'
            '.nav{padding:12px 28px;display:flex;gap:8px;flex-wrap:wrap;'
            'background:#161822;border-bottom:1px solid rgba(255,255,255,0.05)}\n'
            '.nav button{padding:6px 16px;border:1px solid rgba(255,255,255,0.12);'
            'border-radius:6px;background:#1e2030;color:#ccc;cursor:pointer;font-size:0.82rem}\n'
            '.nav button:hover{background:#2a2d45;border-color:rgba(124,138,255,0.4)}\n'
            '.nav button.active{background:rgba(124,138,255,0.2);border-color:#7c8aff;color:#fff}\n'
            'iframe{width:100%;border:none;background:#0f1117}\n'
            '.info{padding:8px 28px;font-size:0.72rem;color:#555}\n'
            '</style></head><body>\n'
            '<header><h1><span>SND@LHC</span> 3D Event Display</h1>\n'
            '<div class="badge">' + flav_name + '</div></header>\n'
            '<div class="nav">' + nav + '</div>\n'
            '<iframe id="viewer" src="' + event_files[0][1] + '"></iframe>\n'
            '<div class="info">circle=SciFi | square=MuFilter US | diamond=MuFilter DS | '
            'Colour=Energy (' + scale_label + ') | Drag to rotate, Scroll to zoom</div>\n'
            '<script>\n'
            "function loadEvent(fn){document.getElementById('viewer').src=fn;"
            "document.querySelectorAll('.nav button').forEach(b=>b.classList.remove('active'));"
            "event.target.classList.add('active')}\n"
            "document.querySelector('.nav button').classList.add('active');\n"
            "function resize(){document.getElementById('viewer').style.height="
            "(window.innerHeight-130)+'px'}\n"
            "resize();window.addEventListener('resize',resize);\n"
            '</script></body></html>')

    with open(os.path.join(flav_dir, 'index.html'), 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='3D Event Display for SND@LHC - X, Y, Z + Energy heatmap')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to .pt file from GNN_s_b.py')
    parser.add_argument('-o', '--output', default='./event_plots_3d',
                        help='Output directory (default: ./event_plots_3d)')
    parser.add_argument('-n', '--max_events', type=int, default=5,
                        help='Max events per flavour (default: 5)')
    parser.add_argument('--linear', action='store_true', default=False,
                        help='Use linear energy scale (default is log)')
    args = parser.parse_args()

    use_log = not args.linear
    os.makedirs(args.output, exist_ok=True)

    print('Loading ' + args.input + ' ...')
    features, flavours = load_data(args.input)
    print('  Events: ' + str(len(features)))
    print('  Energy scale: ' + ('log10' if use_log else 'linear'))

    uniq, counts = np.unique(flavours.numpy(), return_counts=True)
    for f, c in zip(uniq, counts):
        print('  ' + FLAVOUR_LABELS.get(int(f), 'flav ' + str(int(f))).ljust(15) + ': ' + str(c) + ' events')
    print()

    save_all(features, flavours, args.output, args.max_events, use_log)
    print('Done!')


if __name__ == '__main__':
    main()