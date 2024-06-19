import torch
import numpy as np
from scipy.stats import norm

# Neural Net:
# Very simple rn - getting ideas down

def get_translation_matrix(knots, dof):
    # Ax = 0n, x = (a1, b1, c1, d1, ..., an, bn, cn, dn)
    # [Ax]i = (a1 - a2)x^3 + 
    # A[i, j], j % 3 = 0: Degree 3 diff with a1 at 4i, (4+1)i
    system = np.zeros(((len(knots) * 3), (4 * (len(knots) + 1))))
    for i, x in zip(range(0, len(knots)), knots):
        # For every knot, enforce c0 continuity on the two adjoining splines
        system[3 * i, 4 * i] = x ** 3
        system[3 * i, 4 * i+1] = x ** 2
        system[3 * i, 4 * i+2] = x
        system[3 * i, 4 * i+3] = 1
        system[3 * i, 4 *(i + 1)] = - x ** 3
        system[3 * i, 4 *(i + 1)+1] = - x ** 2
        system[3 * i, 4 *(i + 1)+2] = - x
        system[3 * i, 4 *(i + 1)+3] = - 1

    for i, x in zip(range(0, len(knots)), knots):
        # For every knot, enforce c1 continuity on the two adjoining splines
        system[3 * i+1, 4 * i] = 3 * x ** 2
        system[3 * i+1, 4 * i+1] = 2 * x
        system[3 * i+1, 4 * i+2] = 1
        system[3 * i+1, 4 *(i + 1)] = - 3 * x ** 2
        system[3 * i+1, 4 *(i + 1)+1] = - 2 * x
        system[3 * i+1, 4 *(i + 1)+2] = - 1

    for i, x in zip(range(0, len(knots)), knots):
        # For every knot, enforce c2 continuity on the two adjoining splines
        system[3 * i+2, 4 * i] = 6 * x
        system[3 * i+2, 4 * i+1] = 2
        system[3 * i+2, 4 *(i + 1)] = - 6 * x
        system[3 * i+2, 4 *(i + 1)+1] = - 2

    _, _, V = np.linalg.svd(system)

    # Find basis of solution space

    vecs = V[-dof:, :].T

    return vecs, system # Columns are basis


def mid_constrained(strike_knots, mids, dof,  X):
    # Want initialisation such that not only does the knot satisfy our translation requirements, but also f(knot) = mid
    # Assuming that boundary i coincides with mid i
    # Strike knots = [(left  spline index, x value), ...]
    # Get a single non-homegenous solution:

    # Equality constraints
    constraints = []
    # print(X.shape)
    # print(strike_knots)
    for i, x in strike_knots: # strike_knots are knots that correspond to a known strike price. Since C0 continuity already enforced, We only enforce one equality per knot on midprice
        new_arr = np.zeros(X.shape[1])
        new_arr[4 * i] = x ** 3
        new_arr[4 * i + 1] = x ** 2
        new_arr[4 * i + 2] = x
        new_arr[4 * i + 3] = 1
        constraints += [new_arr]
        # print(new_arr, i,x)        
    # print('Xshape', X.shape)
    Xdash = np.vstack([X]+constraints)

    Xinv = np.linalg.pinv(Xdash)
    _, _, V = np.linalg.svd(Xdash)
    basis = V[-(dof - len(strike_knots)):,:].T # Get basis as columns
    b = np.vstack([np.zeros((X.shape[0],1)), np.array(mids).reshape(-1,1)])
    # print(b)
    xstar = Xinv @ b
    return basis, xstar, Xdash



def simple_initialization(mids, strike_knots, dof, basis,  X):
    # Want initialisation such that not only does the knot satisfy our translation requirements, but also f(knot) = mid
    # Assuming that boundary i coincides with mid i
    # Strike knots = [(left spline index, x value), ...]
    # Get a single non-homegenous solution:

    # Equality constraints
    constraints = []
    print(X.shape)
    print(strike_knots)
    for i, x in strike_knots: # strike_knots are knots that correspond to a known strike price. Since C0 continuity already enforced, We only enforce one equality per knot on midprice
        new_arr = np.zeros(X.shape[1])
        new_arr[4 * i] = x ** 3
        new_arr[4 * i + 1] = x ** 2
        new_arr[4 * i + 2] = x
        new_arr[4 * i + 3] = 1
        constraints += [new_arr]
        
    print('Xshape', X.shape)
    Xdash = np.vstack([X]+constraints)

    Xinv = np.linalg.pinv(Xdash)
    print(Xinv.shape, Xdash.shape, X.shape, np.array(mids).reshape(-1,1).shape)
    # Solution for each ai, bi, ci, xi
    b = np.vstack([np.zeros((X.shape[0],1)), np.array(mids).reshape(-1,1)])
    print(b.shape, 'bshape')
    solution = Xinv @ b
    print(basis.shape)
    # Projection onto control points
    return basis.T @ solution


def ind(a,b,c,d):
    return a + 4 * b + 4 * 4 * c + 4 * 4 * 3 * d

def get_dummy():
    """
    Defines a polynomial in terms of a, a^2, a^3, a^4, ab,ac,ad,...
    """

    dummy = torch.zeros(11, (4*4*4*4)).double() 
   
    # x^0
    dummy[0,ind(0,1,0,2)] = 1 / 2.0
    dummy[0,ind(0,0,2,2)] = 1 / 16.0
    dummy[0,ind(0,0,2,1)] = 1 / 4.0
    dummy[0,ind(0,0,0,2)] = 1

    # x^1
    dummy[1,ind(1,2,0,0)] = 3
    dummy[1,ind(0,1,1,2)] = 1 / 4.0
    dummy[1,ind(0,1,1,1)] = 2
    dummy[1,ind(0,0,3,1)] = 1 / 8.0
    dummy[1,ind(0,0,3,0)] = 1 / 4
    dummy[1,ind(0,0,1,1)] = 1

    # x^2
    dummy[2,ind(1,0,1,2)] = 3 / 8
    dummy[2,ind(1,0,1,1)] = 15 / 2
    dummy[2,ind(0,2,0,2)] = 1 / 4
    dummy[2,ind(0,1,2,1)] = 5 / 8
    dummy[2,ind(0,1,2,0)] = 7 / 4
    dummy[2,ind(0,0,4,0)] = 1 / 16
    dummy[2,ind(0,0,2,0)] = 1 / 4

    # x^3
    dummy[3,ind(1,1,0,2)] = 3 / 4
    dummy[3,ind(1,1,0,1)] = 10
    dummy[3,ind(1,0,2,1)] = 7 / 8
    dummy[3,ind(1,0,2,0)] = 19 / 4
    dummy[3,ind(1,0,0,1)] = -1
    dummy[3,ind(0,2,1,1)] = 1
    dummy[3,ind(0,2,1,0)] = 3
    dummy[3,ind(0,1,3,0)] = 3 / 8
    
    # x^4
    dummy[4,ind(2,0,0,2)] = 9 / 16
    dummy[4,ind(2,0,0,1)] = 33 / 4
    dummy[4,ind(1,1,1,1)] = 11 / 4
    dummy[4,ind(1,1,1,0)] = 25 / 2
    dummy[4,ind(1,0,3,0)] = 1 / 2
    dummy[4,ind(1,0,1,0)] = -1 / 2
    dummy[4,ind(0,3,0,1)] = 1 / 2
    dummy[4,ind(0,3,0,0)] = 3 / 2
    dummy[4,ind(0,2,2,0)] = 13 / 16

    # x^5
    dummy[5,ind(2,0,1,1)] = 15 / 8
    dummy[5,ind(2,0,1,0)] = 39 / 4
    dummy[5,ind(1,2,0,1)] = 2
    dummy[5,ind(1,2,0,1)] = 2
    dummy[5,ind(1,2,0,0)] = 8
    dummy[5,ind(1,1,2,0)] = 17 / 8
    dummy[5,ind(0,3,1,0)] = 3 / 4

    # x^6
    dummy[6,ind(2,1,0,1)] = 21 / 8
    dummy[6,ind(2,1,0,0)] = 47 / 4
    dummy[6,ind(2,0,2,0)] = 11 / 8
    dummy[6,ind(2,0,0,0)] = 1 / 4
    dummy[6,ind(1,2,1,0)] = 23 / 8
    dummy[6,ind(0,3,0,0)] = 1 / 4 

    # x^7
    dummy[7,ind(3,0,0,1)] = 9 / 8
    dummy[7,ind(3,0,0,0)] = 21 / 4
    dummy[7,ind(2,1,1,0)] = 29 / 8
    dummy[7,ind(1,3,0,0)] = 5 / 4
    
    # x^8
    dummy[8,ind(3,0,1,0)] = 3 / 2
    dummy[8,ind(2,2,0,0)] = 37 / 16

    # x^9
    dummy[9,ind(3,1,0,0)] = 15 / 8

    # x^10
    dummy[10,ind(4,0,0,0)] = 9 / 16

    return dummy  

def get_int_grad(poly, x0, x1, flag0, flag1):
    # Pred: roots sorted - no overlap between 
    # Flagi = True implies xi is a constant boundary (not a root)
    grad = np.zeros(poly.degree() + 1)
    for i in range(poly.degree() + 1):
        grad[i] += poly(x1) - poly(x0) + flag1 * (poly(x1) * x1 ** i) / poly.deriv()(x1) - flag0 * (poly(x0) * x0 ** i) / poly.deriv()(x0)
    return grad


def get_start_grad(coeffs, x, y):
    # x0 is fixed boundary

    grad = np.zeros(coeffs.shape[0])
    poly = np.polynomial.polynomial.Polynomial(np.flip(coeffs.detach().numpy()))
    roots = np.roots(coeffs.detach.numpy())
    roots = roots.real[abs(roots.imag) < threshold]
    roots = roots[(roots > x) & (roots < y)]
    ints = [x] + list(roots) + [y]
    ints = list(zip(ints[:-1], ints[1:], [1] + [0 for x in range(len(roots) + 1)], [0 for x in range(len(roots)) + 1] + [1]))
    
    for x0, x1, flag0, flag1 in ints:
        grad += 0 if poly((x0 + x1) / 2) > 0 else get_int_grad(poly, x0, x1, flag0, flag1)

    return grad


def loss_on_spline(coeffs, boundaries):
    threshold = 1.0e-5
    x, y = boundaries
    
    # Assume coeffs is of shape (4,)

    # Get coefficient combinations
    # ap = torch.tensor([1, a, a**2, a**3, a**4]).reshape(1,1,1,4)

    # Dummy defines coefficients for a^ib^jc^kd^l
    # map from 4 * 4 * 4 * 4 -> 11;
    # combs needs to be 4 * 4 * 4 * 4

    # Transform coefficients into loss function polynomial

    coeff_pows = coeffs ** torch.tensor([0,1,2,3])
    depth, size = coeff_pows.shape[0], coeff_pows.shape[1]
    combs = coeff_pows[0].reshape(1,1,1,-1)
    shape = [1,1,1,1]
    for i in range(depth - 1, -1, -1):
        shape[i] = -1 
        combs = coeff_pows[i].reshape(*shape) * combs
        shape[i] = 1
    combs = combs.flatten()
    dummy = get_dummy()     
    # dummy of shape (11, 4 * 4 * 4 * 4)
    poly_rev =  (dummy @ combs)
    poly = poly_rev.flip((0)) # Need highest degree first

    # ---------------------------------------------------

    # Get integral of negative parts

    roots = np.roots(poly.detach().numpy())
    roots = roots.real[abs(roots.imag) < threshold]
    roots = torch.tensor(roots)
    new_powers = torch.tensor([i for i in range(poly.shape[0],0,-1)])
    integral = poly / new_powers
    roots, _ = torch.sort(roots)
    roots = roots[(roots > x) & (roots < y)]
    bounds = torch.cat([torch.tensor([x]), roots, torch.tensor([y])])
    pairs = zip(bounds[:-1], bounds[1:])
    vals = torch.sum(torch.tensor(
        [min(
            torch.vdot(integral, torch.tensor(y) ** new_powers) - torch.vdot(integral, torch.tensor(x) ** new_powers),
            torch.tensor(0.0)
        ) for x, y in pairs]))
    return vals

def loss_on_splines(coeffs, boundaries):
    loss = torch.tensor(0).double()
    print(coeffs.shape[0] // 4 - 1)
    print(coeffs.shape) 
    print(boundaries.shape)
    for i in range((coeffs.shape[0] // 4) - 2):
        loss += loss_on_spline(coeffs[4*i:4 * (i+1)], (boundaries[i], boundaries[i+1]))
    return loss

class SmileNet(torch.nn.Module):
    """
    Arguments:
        num_strikes = n
        num_intervals = m - spline boundaries static for now

    Inputs: Log-strike-bid-ask: dim = (2n)
    Layer 1: dim = (2n,k1)
    Layer 2: dim = (k1,k2)
    Layer 3: dim = (k2,k3)
    Layer 4: dim (k3,m + 4)
    Layer 4 structure:
        nodes [0, m) indicate degree of freedom for each spline
        nodes [m, m+3) indicate our remaining degrees of freedom
    Given spline 1 with junction at x with spline 2 and junction at y with spline 0:


    # At x: - 3 equations - 8 unknowns # 5 dof
    # C0 continuity implies a2x^3 + b2x^2 +c2x + d2 = a1x^3 + b1x^3 + c1x + d1
    # C1 continuity implies 3 a2x^2 + 2 b2x +c2 = 3 a1x^2 + 2 b1x +c1
    # C2 continuity implies 6 a2x + b2 = 6 a1x + b1

    # at y:
    # C0 continuity implies a0x^3 + b0x^2 +c0x + d0 = a1x^3 + b1x^3 + c1x + d1
    # C1 continuity implies 3 a0x^2 + 2 b0x +c0 = 3 a1x^2 + 2 b1x +c1
    # C2 continuity implies 6 a0x + b0 = 6 a1x + b1

    6 Equations - 12 unknowns # 6 dof

    Given a set of boundaries [x0, x1, ... xm]
    Singular value decomposition on (4 * (m + 1), 3 * m) matrix
    Pick 4 * (m + 1) - (3 * m) = m + 4 smallest singular values that represent the null space of the matrix
    The output of our Neural network maps onto the m+4 degrees of freedom that we have (our non determined variables)
    Any point in the null space of the matrix is a solution. So our control points control the different eigenvectors of the null space,
    which are obtained from the rows of V.T in the SVD solution X = U S V.T
    These rows represent the impact of this control point on the coefficients of the cubics (which are indexed by column)

    We also want to ensure in each interval [xi, xj] that the cubic has no real roots
    we can do this by ensuring that f(x_i) > 0, j(x_j) > 0, and that for all k s.t f'(k) = 0 we have f(k) > 0
    
    we have that f(x) = ax^3 + bx^2 + cx + d > 0 for each set (a,b,c,d)
    v1 @ sigma + v2 @ sigma + v3 @ sigma ... > 0 for all sigma

    Hence we also need that 
    f'(x) = 3ax^2 + 2bx + c

    # We 


    Next we need to understand the relationship between control points and the loss function. Say we have a loss function, l(A,x), where A is our coefficient vector
    then we get that dl / dci (control i) = dA/dci * dl /dA = normal loss * sum of A where A is in the eigenvector


    """

    def __init__(self, boundaries, strikes, mids):
        super().__init__()
        # Blackbox layers

        dof = len(boundaries) + 4
        strike_knots = []
        count = 0
        for i in range(len(boundaries)):
            if boundaries[i] == strikes[count]:
                strike_knots += [(i,boundaries[count])]
                count += 1
        basis, X = get_translation_matrix(boundaries, dof)
        basis, xstar,xdash = mid_constrained(strike_knots, mids, dof, X)
        self.boundaries = boundaries
        self.system = xdash
        strike_num = strikes.shape[0]
        self.blackbox = torch.nn.Sequential(
            torch.nn.Linear(strike_num * 2, strike_num * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4, strike_num * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4, dof - len(strike_knots)),
        )

        self.blackbox.apply(lambda m: m.weight.data.fill_(0.0) if isinstance(m, torch.nn.Linear) else None)
        self.blackbox.apply(lambda m: m.bias.data.fill_(0.0)  if isinstance(m, torch.nn.Linear) else None)
        # print(xstar.shape, 'xstar')
        # self.blackbox[-1].bias.data = torch.tensor(-1 * xstar.T).float()
        # Assuming stack returns [lambda_1, lambda_2, ... lambda_dof], shape(1 ,dof)
        # basis of shape (4(m+1),dof)
        self.xstar = np.zeros(xstar.shape) # xstar
        self.translate = torch.tensor(basis.T).float() # we get coefficients vector from v @ w.T (assuming w itself is a row vector)



    def forward(self, x):
        control = self.blackbox(x)
        print('Sol ----------------',sol)
        sol = self.translate.T @ control.T # Outputs coefficients
        # if not sanitycheck else sol, (torch.tensor(self.system) @ sol).reshape(-1,1)[int(-x.shape[1] / 2) :]
        return sol 

    def train(self, data, epochs=10, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        for e in range(epochs):
            # Loss and prediction
            pred = self.forward(data)
            loss = loss_on_splines(pred, self.boundaries)
            print(loss)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def plot_output(self):
        pass