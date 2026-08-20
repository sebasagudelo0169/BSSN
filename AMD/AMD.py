import sympy as sp
from sympy.tensor.array import NDimArray
from sympy.matrices import MatrixBase


class AMD():
    def __init__(self, ds, diffs, variables):
        self.ds = ds
        self.diffs = diffs
        self.variables = variables

        self.spatialvariables = self.variables[1:]
        self.t = self.variables[0]
        self.g = self.metric4D()
        self.betha = self.shift()
        self.gamma = self.metric3D()

    def __evaluar_objeto(self, Y):
        """
        Su funcionalidad es recibir una variable y me dice si es un escalar, un vector o una matriz, entonces me retorna 0,1,2 respectivamente
        """
        if isinstance(Y, (NDimArray, MatrixBase)):
            range = Y.shape
            if len(range) == 1:
                return 1
            elif len(range) == 2:
                if range[0] == 1 or range[1] == 1:  # Matrix 1xN or Nx1
                    return 1
                else:
                    return 2
            else:
                print(f"Es un tensor de rango {len(range)}.")
                return 3
        elif isinstance(Y, (sp.Expr, int, float)):
            return 0
        else:
            raise TypeError("Tipo de dato no reconocido.")

    def metric4D(self):
        """
        Extrae la métrica g_{mu nu} desde un elemento de línea ds^2.
        """
        ds2 = sp.expand(self.ds)
        n = len(self.diffs)
        g = sp.Matrix.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    coeff = ds2.coeff(self.diffs[i]**2)
                else:
                    coeff = ds2.coeff(self.diffs[i]*self.diffs[j])
                g[i, j] = sp.simplify(coeff)
        return g

    def shift(self):
        """
        Es el betha con indice arriba
        """
        return (self.g[0, 1:].T)/2

    def metric3D(self):
        return self.g[1:, 1:]

    def lapse(self):
        gammaInv = self.gamma.inv()
        shift_i = sp.simplify(gammaInv * self.betha)
        betaCuadrado = (shift_i.T * self.betha)[0]
        return sp.sqrt(betaCuadrado - self.g[0, 0])

    def christophel3D(self):
        """
        me genera los simbolos de christophel en forma de diccionario donde sus componentes son las claves de cada elemento
        """
        christopheldicci = {}
        gammaInv = self.gamma.inv()
        for i0, i1 in enumerate(self.spatialvariables):
            for j0, j1 in enumerate(self.spatialvariables):
                for k0, k1 in enumerate(self.spatialvariables):
                    christophel = 0
                    for l0, l1 in enumerate(self.spatialvariables):
                        christophel += (1/2)*gammaInv[i0, l0]*(sp.diff(self.gamma[j0, l0], k1) + sp.diff(
                            self.gamma[l0, k0], j1) - sp.diff(self.gamma[j0, k0], l1))
                    christopheldicci[i0, j0, k0] = sp.simplify(christophel)

        return christopheldicci

    def derivadaContravariante3D(self, Y, k=0, i=0):
        """
        Si Y es un vector, k(indice arriba) es la componente a derivar de Y, me devuelve su derivada covariante 3D respecto a t
        """
        bridge = self.__evaluar_objeto(Y)
        if bridge == 0:
            return sp.Array([sp.diff(Y, self.spatialvariables[0]), sp.diff(Y, self.spatialvariables[1]), sp.diff(Y, self.spatialvariables[2])])

        elif bridge == 1:
            t = self.spatialvariables[i]
            SChris = self.christophel3D()
            derivada1 = sp.diff(Y[k], t)
            sumConexion = 0
            for j in range(0, 3):
                sumConexion += SChris[k, i, j]*Y[j]
            return derivada1+sumConexion

    def derivadaCovariante3D(self, Y, k=0, i=0):
        """
        Si Y es un vector, k(indice arriba) es la componente a derivar de Y, me devuelve su derivada covariante 3D respecto a t
        """
        bridge = self.__evaluar_objeto(Y)
        if bridge == 0:
            return sp.Array([sp.diff(Y, self.spatialvariables[0]), sp.diff(Y, self.spatialvariables[1]), sp.diff(Y, self.spatialvariables[2])])

        elif bridge == 1:
            t = self.spatialvariables[i]
            SChris = self.christophel3D()
            derivada1 = sp.diff(Y[k], t)
            sumConexion = 0
            for j in range(0, 3):
                sumConexion += SChris[j, i, k]*Y[j]
            return derivada1-sumConexion

    def extrinsicCurvature(self):
        K = sp.Matrix.zeros(3)
        for i in range(0, 3):
            for j in range(0, 3):
                kij = (1/(2*self.lapse()))*(self.derivadaCovariante3D(self.betha, i, j) +
                                            self.derivadaCovariante3D(self.betha, j, i) - sp.diff(self.gamma[i, j], self.t))
                K[i, j] = kij

        return K
