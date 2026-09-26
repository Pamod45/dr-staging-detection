"""HTML diagrams rendered with st.html. Colours match .streamlit/config.toml. Every rule is
scoped under .dr-diagram, because st.html puts the CSS on the page itself."""

BG_CARD = "#1A242B"
TEXT = "#E3EAEE"
MUTED = "#8A99A3"
TUNED = "#2E6F8E"
HEAD = "#4FA3C7"

_CSS = f"""
<style>
  .dr-diagram {{ color: {TEXT}; }}
  .dr-diagram .phases {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .dr-diagram .phase {{ background: {BG_CARD}; border-radius: 8px; padding: 12px 14px; flex: 1 1 280px; }}
  .dr-diagram .phase h4 {{ margin: 0 0 6px; font-size: 15px; }}
  .dr-diagram .phase p {{ margin: 3px 0; font-size: 13px; color: {MUTED}; }}
  .dr-diagram .phase p b {{ color: {TEXT}; font-weight: 600; }}
  .dr-diagram .bar {{ display: flex; height: 26px; border-radius: 6px; overflow: hidden; margin: 14px 0 4px;
          position: relative; font-size: 12px; }}
  .dr-diagram .bar div {{ display: flex; align-items: center; justify-content: center; color: {TEXT}; }}
  .dr-diagram .mark {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: #F2C14E; }}
  .dr-diagram .ticks {{ display: flex; justify-content: space-between; font-size: 11px; color: {MUTED}; }}
</style>
"""


def phases(p1_epochs: int, p1_lr: float, epochs_run: int, best: int, p2_lr: float,
           counts: dict, rlr_factor: float, rlr_patience: int, es_patience: int) -> str:
    p2_epochs = epochs_run - p1_epochs
    w1 = 100 * p1_epochs / epochs_run
    best_pos = 100 * (best - 0.5) / epochs_run
    card1 = (f'<div class="phase" style="border-top:4px solid {HEAD}"><h4>Phase 1: train the '
             f'new layers</h4><p>Epochs <b>1 to {p1_epochs}</b>, learning rate '
             f'<b>{p1_lr:g}</b></p><p>Backbone <b>frozen</b>, all {counts["backbone_layers"]} '
             f'layers</p><p>Trainable parameters <b>{counts["phase1"]:,}</b> of '
             f'{counts["total"]:,}</p><p>Why: random new layers would send large, noisy '
             f'updates into the pretrained backbone and damage it.</p></div>')
    card2 = (f'<div class="phase" style="border-top:4px solid {TUNED}"><h4>Phase 2: fine-tune '
             f'the upper backbone</h4><p>Epochs <b>{p1_epochs + 1} to {epochs_run}</b>, '
             f'learning rate <b>{p2_lr:g}</b>, cut to {rlr_factor:g}x after {rlr_patience} '
             f'epochs without improvement</p><p>Upper <b>{counts["unfrozen_layers"]} of '
             f'{counts["backbone_layers"]}</b> backbone layers unfrozen</p><p>Trainable '
             f'parameters <b>{counts["phase2"]:,}</b></p><p>Stopped after {es_patience} epochs '
             f'without improvement; weights from epoch <b>{best}</b> kept.</p></div>')
    bar = (f'<div class="bar"><div style="width:{w1}%;background:{HEAD}">Phase 1</div>'
           f'<div style="width:{100 - w1}%;background:{TUNED}">Phase 2, {p2_epochs} epochs</div>'
           f'<div class="mark" style="left:{best_pos}%" title="best epoch {best}"></div></div>'
           f'<div class="ticks"><span>epoch 1</span><span>best epoch {best} (yellow line)</span>'
           f'<span>epoch {epochs_run}</span></div>')
    return _CSS + f'<div class="dr-diagram"><div class="phases">{card1}{card2}</div>{bar}</div>'
