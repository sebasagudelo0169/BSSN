from AMD import AMD
import sympy as sp
from IPython.display import display

dt, dr, dtheta, dphi = sp.symbols(
    r'dt dr d\theta d\phi'
)

# Coordenadas
t, r, theta, phi, M = sp.symbols(
    r't r \theta \phi M',
    positive=True
)
diffs0 = [dt, dr, dtheta, dphi]
variables0 = [t, r, theta, phi]
# Factor
f = 1 - 2*M/r

# ds^2
ds2 = (
    -f*dt**2
    + dr**2/f
    + r**2*dtheta**2
    + r**2*sp.sin(theta)**2*dphi**2
)
display(ds2)
# if __name__ == '__main__':
#     general =
