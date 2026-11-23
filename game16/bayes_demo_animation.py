"""Анимация: байесовское обновление уверенности по мере поступления данных.

Модель та же, что в MCMC_demo.ipynb:  y = k x + b + шум N(0, sigma^2).
На каждом шаге добавляется одно измерение, и апостериор

    p(k, b | данные)  ~  p(данные | k, b) * p(k, b)

пересчитывается на сетке; он же становится приором для следующего шага.
Запуск:  python3 bayes_demo_animation.py   ->  bayes_demo_updating.mp4
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------------------------------------------ данные
SEED = 42
N, K_TRUE, B_TRUE, SIGMA = 40, 2.5, -1.0, 0.8
rng = np.random.default_rng(SEED)
x = np.sort(rng.uniform(0, 10, N))
y = K_TRUE*x + B_TRUE + rng.normal(0, SIGMA, N)

# приор: равномерный в прямоугольнике (он же диапазон осей)
K_LO, K_HI, B_LO, B_HI = -2.0, 7.0, -10.0, 8.0
kk = np.linspace(K_LO, K_HI, 400)
bb = np.linspace(B_LO, B_HI, 400)
KK, BB = np.meshgrid(kk, bb)

CHI = np.zeros_like(KK)               # накапливается chi^2 по точкам
x_plot = np.linspace(-0.3, 10.3, 100)

# ------------------------------------------------------------------ фигура
fig = plt.figure(figsize=(12.6, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95], width_ratios=[1.25, 1.0, 1.05],
                      hspace=0.45, wspace=0.28)
ax_data = fig.add_subplot(gs[0, :2])
ax_post = fig.add_subplot(gs[0, 2])
ax_pk = fig.add_subplot(gs[1, 0])
ax_pb = fig.add_subplot(gs[1, 1])
ax_txt = fig.add_subplot(gs[1, 2])
ax_txt.axis('off')

# --- левая панель: данные и предсказание апостериора
ax_data.plot(x_plot, K_TRUE*x_plot + B_TRUE, ls='--', lw=1.3, color='0.35')
ax_data.set_xlim(-0.5, 10.5)
ax_data.set_ylim(y.min() - 4*SIGMA, y.max() + 4*SIGMA)
ax_data.set_xlabel('$x$')
ax_data.set_ylabel('$y$')
ax_data.grid(True, color='0.9', lw=0.6)
ax_data.legend(handles=[
    Line2D([], [], ls='--', lw=1.3, color='0.35', label='истина'),
    Line2D([], [], ls='none', marker='o', ms=5, color='0.15', label='измерения'),
    Line2D([], [], lw=2, color='#eb6834', label=r'$\bar k x+\bar b$'),
    Patch(facecolor='#eb6834', alpha=0.25, label=r'$\pm1\sigma$ предсказания'),
], frameon=False, loc='upper left', ncol=2, fontsize=9)
for sp in ('top', 'right'):
    ax_data.spines[sp].set_visible(False)

fit_line, = ax_data.plot([], [], lw=2, color='#eb6834', zorder=5)

# --- правая верхняя: апостериор в плоскости (k, b)
im = ax_post.imshow(np.ones_like(KK), extent=[K_LO, K_HI, B_LO, B_HI], origin='lower',
                    aspect='auto', cmap='Blues', vmin=0.0, vmax=1.0, interpolation='bilinear')
ax_post.plot(K_TRUE, B_TRUE, '*', ms=13, color='0.15', zorder=5)
mean_dot, = ax_post.plot([], [], 'o', ms=7, mfc='none', mec='#eb6834', mew=1.8, zorder=6)
like_lo, = ax_post.plot([], [], ls=':', lw=1.1, color='0.35', zorder=4)
like_hi, = ax_post.plot([], [], ls=':', lw=1.1, color='0.35', zorder=4)
prior_note = ax_post.text(0.5, 0.5, 'равномерный приор\nизмерений нет', transform=ax_post.transAxes,
                          ha='center', va='center', color='0.4', fontsize=10)
ax_post.set_xlabel('$k$')
ax_post.set_ylabel('$b$')
ax_post.set_title('апостериор $p(k,b\\,|\\,$данные)')
for sp in ('top', 'right'):
    ax_post.spines[sp].set_visible(False)

# --- нижние: маргинальные распределения
ax_pk.set_xlim(1.0, 3.5)
ax_pb.set_xlim(-2.6, 0.9)
for ax, name, title in ((ax_pk, '$k$', 'сколько мы знаем о $k$'),
                        (ax_pb, '$b$', 'сколько мы знаем о $b$')):
    ax.set_xlabel(name)
    ax.set_ylabel('плотность (норм.)')
    ax.set_title(title)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.grid(True, color='0.9', lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
ax_pk.axvline(K_TRUE, color='0.25', lw=1.2, ls='--')
ax_pb.axvline(B_TRUE, color='0.25', lw=1.2, ls='--')
curve_k, = ax_pk.plot([], [], lw=2, color='#2a78d6')
curve_b, = ax_pb.plot([], [], lw=2, color='#2a78d6')
txt_k = ax_pk.text(0.03, 0.95, '', transform=ax_pk.transAxes, fontsize=9, va='top')
txt_b = ax_pb.text(0.03, 0.95, '', transform=ax_pb.transAxes, fontsize=9, va='top')

# --- правая нижняя: формула и текущие числа
step_txt = ax_txt.text(0.0, 0.96, '', transform=ax_txt.transAxes, fontsize=12, va='top',
                       fontweight='bold')
ax_txt.text(0.0, 0.74, r'$p(\theta\,|\,\mathbf{y})=\dfrac{p(\mathbf{y}\,|\,\theta)\,p(\theta)}{p(\mathbf{y})}$'
                       '\n\n' r'$\propto\ \mathcal{L}(\theta)\cdot$ приор',
            transform=ax_txt.transAxes, fontsize=14, va='top')
mean_txt = ax_txt.text(0.0, 0.30, '', transform=ax_txt.transAxes, fontsize=10.5, va='top',
                       color='0.15', linespacing=1.7)
hint_txt = ax_txt.text(0.0, 0.02, 'апостериор → приор следующего шага', transform=ax_txt.transAxes,
                       fontsize=9, va='bottom', style='italic', color='0.45')
suptitle = fig.suptitle('', fontsize=13)


# ------------------------------------------------------------------ шаги
state = {'err': None, 'band': None}


def update(n):
    global CHI
    if n > 0:
        CHI = CHI + ((y[n - 1] - KK*x[n - 1] - BB)/SIGMA)**2

    P = np.exp(-0.5*(CHI - CHI.min())) if n > 0 else np.ones_like(CHI)
    P /= P.max()
    if n == 0:
        im.set_data(np.zeros_like(KK))
        prior_note.set_visible(True)
    else:
        im.set_data(P)
        prior_note.set_visible(False)

    # данные (перерисовываем, чтобы росли планки ошибок)
    if state['err'] is not None:
        state['err'].remove()
    state['err'] = ax_data.errorbar(x[:n], y[:n], yerr=SIGMA, fmt='o', ms=5, lw=1,
                                    capsize=0, color='0.15', zorder=4)

    # предсказание апостериора: средняя прямая и полоса +-1 sigma
    if state['band'] is not None:
        state['band'].remove()
        state['band'] = None
    if n >= 2:
        A = np.column_stack([x[:n], np.ones(n)])
        cov = SIGMA**2*np.linalg.pinv(A.T @ A)
        beta = np.linalg.pinv(A.T @ A) @ (A.T @ y[:n])
        Ak = np.column_stack([x_plot, np.ones_like(x_plot)])
        sd = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Ak, cov, Ak), 0.0))
        mu = beta[0]*x_plot + beta[1]
        fit_line.set_data(x_plot, mu)
        state['band'] = ax_data.fill_between(x_plot, mu - sd, mu + sd,
                                             color='#eb6834', alpha=0.25, lw=0, zorder=3)
        mean_dot.set_data([beta[0]], [beta[1]])
        txt_k.set_text(f'$k={beta[0]:.3f}\\pm{np.sqrt(cov[0, 0]):.3f}$')
        txt_b.set_text(f'$b={beta[1]:.3f}\\pm{np.sqrt(cov[1, 1]):.3f}$')
        mean_txt.set_text(f'$k = {beta[0]:.3f} \\pm {np.sqrt(cov[0, 0]):.3f}$\n'
                          f'$b = {beta[1]:.3f} \\pm {np.sqrt(cov[1, 1]):.3f}$')
        # правдоподобие последней точки: |y_i - k x_i - b| = sigma
        xi, yi = x[n - 1], y[n - 1]
        like_lo.set_data(kk, yi - kk*xi - SIGMA)
        like_hi.set_data(kk, yi - kk*xi + SIGMA)
    else:
        fit_line.set_data([], [])
        mean_dot.set_data([], [])
        like_lo.set_data([], [])
        like_hi.set_data([], [])
        txt_k.set_text('')
        txt_b.set_text('')
        mean_txt.set_text('приор: измерений ещё нет,\nраспределение равномерное')

    curve_k.set_data(kk, P.sum(axis=0)/P.sum(axis=0).max())
    curve_b.set_data(bb, P.sum(axis=1)/P.sum(axis=1).max())

    step_txt.set_text(f'N = {n}')
    suptitle.set_text(f'Байесовское обновление: {n} измерений из {N}' if n
                      else 'Байесовское обновление: один приор, данных нет')


frames = [0]*12 + [n for n in range(1, N + 1) for _ in range(5)] + [N]*18
writer = FFMpegWriter(fps=24, bitrate=3000)
with writer.saving(fig, 'bayes_demo_updating.mp4', dpi=140):
    for n in frames:
        update(n)
        writer.grab_frame()
print('сохранено: bayes_demo_updating.mp4, кадров', len(frames))
