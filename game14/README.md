### Основные уравнения метода ПВД и необходимые соотношения

\begin{aligned}
D_x &= \cos\alpha \cos\delta \\
D_y &= \sin\alpha \cos\delta \\
D_z &= \sin\delta
\end{aligned}

\begin{aligned}
T_x &= -\sin\psi\,\sin\alpha - \cos\psi\,\sin\delta\,\cos\alpha, \\
T_y &= \sin\psi\,\cos\alpha - \cos\psi\,\sin\delta\,\sin\alpha, \\
T_z &= \cos\psi\,\cos\delta.
\end{aligned}

\begin{gather*}
\mathbf{D}\cdot\mathbf{D} = 1, \qquad
\mathbf{D}\cdot\dot{\mathbf{D}} = 0, \qquad
\dot{\mathbf{D}}\cdot\dot{\mathbf{D}} + \mathbf{D}\cdot\ddot{\mathbf{D}} = 0
\\[6pt]
\mu^2 = \bigl(\dot D\bigr)^2 = \dot D_x^2 + \dot D_y^2 + \dot D_z^2
\\[6pt]
\mu\cos\psi = \frac{\dot D_z}{\left(D_x^2 + D_y^2\right)^{1/2}}, \qquad
\mu\sin\psi = \frac{D_x \dot D_y - D_y \dot D_x}{\left(D_x^2 + D_y^2\right)^{1/2}}
\\[6pt]
\mu^3 \varkappa =
\begin{vmatrix}
D_x & D_y & D_z \\
\dot D_x & \dot D_y & \dot D_z \\
\ddot D_x & \ddot D_y & \ddot D_z
\end{vmatrix}
\end{gather*}

#### Уравнения метода ПВД
\begin{gather*}
d\,\mathbf{D} = \mathbf{r} - \mathbf{g}
\\[4pt]
\dot d\,\mathbf{D} + d\mu\,\mathbf{T} = \dot{\mathbf{r}} - \dot{\mathbf{g}}
\\[4pt]
\left(\ddot d - \mu^2 d + \kappa^2 r^{-3} d\right)\mathbf{D}
+ \left(2\mu\dot d + d\dot\mu\right)\mathbf{T}
+ \varkappa\mu^2 d\,\mathbf{M}
= -\kappa^2 r^{-3}\,\mathbf{g} - \ddot{\mathbf{g}}
\end{gather*}

#### Решение уравнений

Здесь $\mathbf{M} = \mathbf{D}\times\mathbf{T}$ (правая тройка $\mathbf{D}, \mathbf{T}, \mathbf{M}$), $(\mathbf{a}\,\mathbf{b}\,\mathbf{c}) = (\mathbf{a}\times\mathbf{b})\cdot\mathbf{c}$ - смешанное произведение, так что $(\mathbf{T}\,\mathbf{D}\,\mathbf{M}) = -1$. Третье уравнение - проекция уравнения для $\ddot{(d\mathbf{D})}$ на $\mathbf{M}$.

\begin{gather*}
\mathbf{r} = \mathbf{g} + d\,\mathbf{D}
\\[4pt]
r^2 = g^2 + d^2 + 2\,(\mathbf{g}\cdot\mathbf{D})\,d = C_0 + 2C_1 d + d^2
\\[4pt]
-\varkappa\mu^2 d\,\underbrace{(\mathbf{T}\,\mathbf{D}\,\mathbf{M})}_{-1}
= \varkappa\mu^2 d
= \kappa^2\,(\mathbf{T}\,\mathbf{D}\,\mathbf{g})\,r^{-3} + (\mathbf{T}\,\mathbf{D}\,\ddot{\mathbf{g}})
\\[4pt]
C d = C_3\, r^{-3} + C_2
\end{gather*}

Константы:

\begin{aligned}
C_0 &= g^2 = \mathbf{g}\cdot\mathbf{g}, \\
C_1 &= \mathbf{g}\cdot\mathbf{D}, \\
C   &= \varkappa\mu^2, \\
C_2 &= (\mathbf{T}\,\mathbf{D}\,\ddot{\mathbf{g}}), \\
C_3 &= \kappa^2\,(\mathbf{T}\,\mathbf{D}\,\mathbf{g}).
\end{aligned}

Еще раз уравнения:

\begin{gather*}
r^2 = C_0 + 2C_1 d + d^2
\\[4pt]
C d = C_3\, r^{-3} + C_2
\end{gather*}
