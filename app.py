import colorsys
import io
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path


def color_palette(n, rng, saturation=0.65):
    base_hue = rng.uniform(0.95, 0.99)
    colors = []
    for i in range(n):
        hue = (base_hue + rng.uniform(-0.025, 0.025)) % 1
        if i % 3 == 0:
            lightness = rng.uniform(0.72, 0.84)
        elif i % 3 == 1:
            lightness = rng.uniform(0.47, 0.62)
        else:
            lightness = rng.uniform(0.28, 0.40)
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def blob(rng, center=(0, 0), radius=1.0, wobble=0.22,
         stretch=(1, 1), rotation=0, points=500):
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    wave = np.zeros_like(theta)
    for frequency in range(2, 6):
        phase = rng.uniform(0, 2 * np.pi)
        wave += np.sin(frequency * theta + phase) / frequency**1.5
    wave /= max(np.max(np.abs(wave)), 1e-12)
    r = radius * (1 + wobble * wave)
    x = r * np.cos(theta) * stretch[0]
    y = r * np.sin(theta) * stretch[1]
    c, s = np.cos(rotation), np.sin(rotation)
    return np.column_stack((c*x - s*y + center[0],
                            s*x + c*y + center[1]))


def draw(n_shapes=8, radius=2.5, wobble=0.22,
         alpha=0.72, saturation=0.65, seed=42,
         background='#F5F1E9', title='Generative Poster'):
    if not isinstance(n_shapes, (int, np.integer)) or not 6 <= n_shapes <= 10:
        raise ValueError('Layers must be an integer between 6 and 10.')
    if radius <= 0 or not 0 <= wobble < 1:
        raise ValueError('Use radius > 0 and 0 <= wobble < 1.')
    if not 0 <= alpha <= 1 or not 0 <= saturation <= 1:
        raise ValueError('Alpha and saturation must be between 0 and 1.')
    rng = np.random.default_rng(seed)
    colors = color_palette(n_shapes, rng, saturation)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 8), ylim=(0, 10), aspect='equal')
    ax.axis('off')
    for i, scale in enumerate(np.linspace(1.15, 0.42, n_shapes)):
        t = i / (n_shapes - 1)
        center = (
            4 + 0.7*np.sin(t*2*np.pi) + rng.uniform(-0.25, 0.25),
            5.2 + np.cos(t*2*np.pi) + rng.uniform(-0.25, 0.25))
        vertices = blob(
            rng, center=center, radius=radius*scale, wobble=wobble,
            stretch=(rng.uniform(0.85, 1.15), rng.uniform(0.95, 1.25)),
            rotation=rng.uniform(0, 2*np.pi))
        ax.add_patch(Polygon(vertices, closed=True, facecolor=colors[i],
                             edgecolor='none', alpha=alpha, zorder=i+1))
    if title:
        ax.text(0.55, 9.25, title, fontsize=30, weight='bold',
                color='#202020', va='top', zorder=20)
        ax.text(7.45, 0.65, f'{n_shapes:02d} LAYERS', fontsize=9,
                color='#202020', ha='right', zorder=20)
    return fig, ax


def add_vintage_texture(fig, ax, grain=0.10,
                        petal_texture=0.16, vignette=0.10, seed=123):
    rng = np.random.default_rng(seed)
    for artist in list(ax.images) + list(ax.patches):
        if artist.get_gid() == 'vintage_texture':
            artist.remove()
    petals = [p for p in ax.patches if isinstance(p, Polygon)]
    for petal in petals:
        vertices = petal.get_xy()
        xmin, ymin = vertices.min(axis=0)
        xmax, ymax = vertices.max(axis=0)
        width, height = xmax-xmin, ymax-ymin
        cx = (xmin+xmax)/2
        for j in range(22):
            spread = (j/21 - 0.5)*1.8
            drift = rng.uniform(-0.04, 0.04)*width
            points = [
                (cx + spread*width*0.10, ymin-height*0.08),
                (cx + spread*width*0.65 + drift, ymin+height*0.28),
                (cx + spread*width*0.55 + drift, ymin+height*0.72),
                (cx + spread*width*0.30, ymax+height*0.08)]
            path = Path(points, [Path.MOVETO, Path.CURVE4,
                                 Path.CURVE4, Path.CURVE4])
            line = PathPatch(
                path, facecolor='none',
                edgecolor='#FFF0DF' if j % 3 else '#713645',
                linewidth=rng.uniform(0.35, 0.75), alpha=petal_texture,
                zorder=petal.get_zorder()+0.1)
            line.set_clip_path(petal)
            line.set_gid('vintage_texture')
            ax.add_patch(line)
    h, w = 1250, 1000
    noise = rng.normal(0, 1, (h, w))
    paper = np.empty((h, w, 4))
    light = np.array([1.0, 0.97, 0.90])
    dark = np.array([0.29, 0.20, 0.17])
    paper[..., :3] = np.where((noise > 0)[..., None], light, dark)
    paper[..., 3] = np.clip(np.abs(noise)*grain, 0, 0.30)
    texture = ax.imshow(paper, extent=(0, 8, 0, 10), origin='lower',
                        interpolation='bilinear', zorder=15)
    texture.set_gid('vintage_texture')
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    edge = np.clip((xx**2 + yy**2 - 0.35)/1.65, 0, 1)
    shading = np.zeros((h, w, 4))
    shading[..., :3] = [0.40, 0.27, 0.18]
    shading[..., 3] = edge**1.5*vignette
    shade = ax.imshow(shading, extent=(0, 8, 0, 10), origin='lower',
                      interpolation='bilinear', zorder=16)
    shade.set_gid('vintage_texture')
    return fig, ax


@st.cache_resource
def rendering_lock():
    # Matplotlib is shared across Streamlit sessions.
    return threading.RLock()


def main():
    st.set_page_config(page_title='Generative Poster', layout='centered')
    st.title('Generative Poster')
    st.caption('Adjust the sidebar controls to explore rose colors, flowing '
               'petals, and vintage paper textures. Download your poster as a PNG.')
    with st.sidebar:
        st.header('Poster controls')
        layers = st.slider('Layers', 6, 10, 8, 1)
        radius = st.slider('Radius', 1.5, 3.5, 2.5, 0.05)
        wobble = st.slider('Wobble', 0.00, 0.60, 0.22, 0.01)
        alpha = st.slider('Opacity (alpha)', 0.20, 1.00, 0.72, 0.01)
        saturation = st.slider('Rose color saturation', 0.20, 0.90, 0.65, 0.01)
        seed = st.number_input('Poster seed', min_value=0, max_value=9999,
                               value=42, step=1)
        st.subheader('Vintage texture')
        grain = st.slider('Paper grain', 0.00, 0.25, 0.10, 0.01)
        petal_texture = st.slider('Petal line texture', 0.00, 0.35, 0.16, 0.01)
        vignette = st.slider('Vignette', 0.00, 0.30, 0.10, 0.01)

    with rendering_lock():
        fig, ax = draw(n_shapes=layers, radius=radius, wobble=wobble,
                       alpha=alpha, saturation=saturation, seed=int(seed))
        try:
            add_vintage_texture(fig, ax, grain=grain,
                                petal_texture=petal_texture,
                                vignette=vignette, seed=123)
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format='png', dpi=300,
                            facecolor=fig.get_facecolor(), transparent=False)
                png_bytes = buffer.getvalue()
            st.pyplot(fig)
        finally:
            plt.close(fig)
    st.download_button('Download PNG (300 dpi)', data=png_bytes,
                       file_name='rose_vintage_poster.png', mime='image/png')
    st.caption('2400 × 3000 pixels · 8 × 10 inches at 300 dpi')


if __name__ == '__main__':
    main()
