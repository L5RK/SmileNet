import torch
import numpy as np
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt

# Neural Net:
# Very simple rn - getting ideas down

torch.set_default_dtype(torch.double)


def get_system_matrix(num_knots):
    # Ax = 0n, x = (a1, b1, c1, d1, ..., an, bn, cn, dn)
    # [Ax]i = (a1 - a2)x^3 + 
    # A[i, j], j % 3 = 0: Degree 3 diff with a1 at 4i, (4+1)i
    # -> add wings => 4 extra constraints: a0, b0, an, bn = 0 => remove 4 dof => need at least 4 knots
    system = torch.zeros(((num_knots * 3 + 4), (4 * (num_knots + 1)))).double()

    for i in range(0, num_knots, 1):
        # For every knot, enforce c0 continuity on the two adjoining splines
        system[3 * i,   (4*i):(4*i)+8] = torch.tensor([1,1,1,1,-1,-1,-1,-1]) # C0
        system[3 * i+1, (4* i):(4*i)+8] = torch.tensor([0,1,2,3,0,-1,-2,-3]) # C1
        system[3 * i+2, (4* i):(4*i)+8] = torch.tensor([0,0,2,6,0,0,-2,-6]) # C2
    # system.requires_grad_()
    system[-4, 0:4] = torch.tensor([0,1,2,3])
    system[-3, -4:] = torch.tensor([0,1,2,3])
    system[-2, 0:4] = torch.tensor([0,0,2,6])
    system[-1, -4:] = torch.tensor([0,0,2,6])
    

    return system

def get_basis(system, knots, seeded_vecs, k_start, k_end):
    print(knots, k_start, k_end)
    p1 = torch.tensor([0,1,2,3])
    p2 = torch.tensor([0,0,1,2])
    p3 = torch.tensor([0,0,0,1])
    knots = knots.reshape(-1,1)
    kedge = torch.hstack([k_start, k_end]).reshape(-1,1)
    kp1 = (knots ** p1).repeat(1,knots.shape[0] + 1)
    kp2 = (knots ** p2).repeat(1,knots.shape[0] + 1)
    kp3 = (knots ** p3).repeat(1,knots.shape[0] + 1)
    kp4 = (kedge ** p2).repeat(1, knots.shape[0] + 1)
    kp5 = (kedge ** p3).repeat(1, knots.shape[0] + 1)
    knot_powers = torch.hstack([kp1,kp2,kp3]).reshape(-1, (knots.shape[0] + 1) * 4)
    new_sys = knot_powers * system[:-4,:]
    print(system[-2:,:].shape, kp4.shape, kp5.shape, torch.vstack([kp4, kp5]).shape)
    new_sys = torch.vstack([new_sys, torch.vstack([kp4, kp5]) * system[-4:,:]])
    # new_sys = torch.vstack([new_sys, system[:,:]])
    print(new_sys.round())
    Q, R = torch.linalg.qr(new_sys.T)
    print("QR")
    print(Q.round())
    print(R.round())
    print(torch.max(torch.abs(Q @ R - new_sys.T)))
    # We want to orthogonalize our basis after projecting from Q to ensure that
    # we can span the space with a lower magnitude of coefficient
    # to minimize rounding error
    # We will do this now
    # projected_basis = seeded_vecs - Q @ (Q.T @ seeded_vecs)
    # print(seeded_vecs.shape, Q.shape)
    basis = seeded_vecs[:,[0]] - Q @ (Q.T @ seeded_vecs[:,[0]])
    basis = basis / torch.linalg.norm(basis)
    for i in range(1, seeded_vecs.shape[1]):
        # print((Q @ (Q.T @ seeded_vecs[:,i])).shape)
        wb = seeded_vecs[:,[i]] - Q @ (Q.T @ seeded_vecs[:,[i]])
        wb = wb -  basis @ (basis.T @ wb)
        wb = wb / torch.linalg.norm(wb)
        basis = torch.hstack([basis, wb])
    print("--------------")
    print((basis.T @ basis).round())
    print("--------")
    # print("in basis")
    return basis, new_sys

def get_vector_seed(num_knots):
    dof = num_knots  + 4 - 4 # Subject to change -> Linear wings -4 for linear wings
    # Dimensionality of null space is dof 
    # need dof vectors to span it
    xs = torch.normal(0, torch.ones((4 * (num_knots + 1)),dof))
    xs = (xs / xs.norm(dim=0))
    return xs

def get_knots(strikemin, strikemax, diffs):
    # Gets knots such that they are strictly increasing from series of n control points    
    n = torch.zeros(diffs.shape)
    t = 0
    for i in range(diffs.shape[1]):
        t += diffs[0,i]
        n[0,i] = t
    return n

def get_xstar(basis, knots, strikes, mids, ini_sys, spreads=None):
    # Pred - knot n is before strike n
    # Pred - strikes are sorted increasing
    # pred - knots are sorted increasing
    # print("going in ")
    # print(basis.shape, knots.shape, mids.shape, ini_sys.shape)
    # 0 , 1, 2, 3
    if spreads is None:
        spreads = torch.ones(mids.shape)
    mids = torch.sqrt(mids)
    mids =  mids 
    spreads = torch.diag(torch.sqrt(1/spreads))
    eqs = torch.zeros(strikes.shape[0], 4 * (knots.shape[0] + 1)).double()
    inds_offset = torch.arange(knots.shape[0])
    knotsR = (torch.hstack([knots, strikes[[-1]] + 1.0])).repeat((strikes.shape[0],1)).T
    # print(inds_offset)
    inds = torch.where((knotsR[inds_offset] < strikes) & (knotsR[inds_offset+1] > strikes))[0]
    inds = torch.hstack([inds, torch.ones(strikes.shape[0] - inds.shape[0]).type(torch.int) * knots.shape[0]]).type(torch.int)
    powers = torch.tensor([0,1,2,3])
    strike_powers = (strikes.reshape(-1,1) ** powers).repeat(1, knots.shape[0] + 1) # number polys = number knots + 1
    for i in range(inds.shape[0]):
        eqs[i,4*inds[i]:4*(inds[i]+1)] = 1

    eqs = strike_powers * eqs
    bsol  = torch.linalg.lstsq(spreads @ (eqs @ basis), spreads @ mids.reshape(-1,1))[0]
    sol = basis @ bsol
    polys = [np.polynomial.polynomial.Polynomial(list(sol[:,0].detach().numpy()[4*i:4*i+4])) for i in range((sol.shape[0])//4)]
    plot_polys([(lambda x, p=poly: p(x)**2) for poly in polys], torch.hstack([torch.tensor(-0.5), knots, torch.tensor(0.25)]), 0, 'before', extra_points=zip(strikes, mids**2))
    return bsol

def process_knot_out(knot_out, lowest):
    ret = torch.zeros(knot_out.shape).reshape(-1) + lowest.item()
    for i in range(knot_out.shape[0]):
        ret[i:] = ret[i:] + knot_out.reshape(-1)[i]
    ret = ret.reshape(knot_out.shape)
    return ret

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

    1 poly: 2 boundaries
    then 2 

    """

    def __init__(self, num_knots, strikes, mids):
        super().__init__()
        # Blackbox layers
        # For now assume fixed strikes for each input
        dof = num_knots + 4  - 4# Linear wings -> -4 degrees of freedom
        strike_knots = []
        count = 0
        # initial_knots = (torch.arange(num_knots)) * (strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1)+ (strikes[0] + strikes[1]) / 2
        strikes = torch.exp(strikes)
        initial_knots = (torch.arange(num_knots + 2)) * ((strikes[-10] + strikes[-1])/2 - (strikes[10] + strikes[0]) / 2) / ((num_knots + 2)- 1)+ (strikes[0] + strikes[10]) / 2
        strikes = torch.log(strikes)
        initial_knots = torch.log(initial_knots)
        # initial_knots = strikes[[i * strikes.shape[0] // (num_knots+2) for i in range(1,num_knots+1)]]
        self.strike_low = strikes[0]
        self.strike_high = strikes[-1]
        self.seeded_vecs = get_vector_seed(num_knots)
        self.base_system = get_system_matrix(num_knots)
        print(initial_knots.shape)
        print(self.base_system)
        init_basis, ini_sys = get_basis(self.base_system, initial_knots[1:-1], self.seeded_vecs, initial_knots[[0]], initial_knots[[1]])
        print(ini_sys)
        xstar = get_xstar(init_basis, initial_knots[1:-1], strikes, mids, ini_sys)
        strike_num = strikes.shape[0]

        self.blackbox = torch.nn.Sequential(
            torch.nn.Linear(strike_num * 3 + 3, strike_num * 4 + 3), # 2*strikes for bid, ask + 1*strikes for strike pos + T for time til maturity + R for interest rate + S for current price
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4 + 3, strike_num * 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 8, strike_num * 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 16, dof + (num_knots + 2)) #- len(strike_knots)),
        )
        self.relu = torch.nn.ReLU()
        self.knot_skeleton = initial_knots
        for i in self.blackbox:
            if not isinstance(i, torch.nn.LeakyReLU):
                i.weight = torch.nn.Parameter(torch.zeros(i.weight.shape).double() / 100)
                i.bias = torch.nn.Parameter(torch.zeros(i.bias.shape).double() / 100)
        new_bias = torch.zeros(self.blackbox[-1].bias.shape)
        # new_bias[:dof] = xstar[:,0]
        # i_bias = torch.ones(num_knots).double() 
        # i_bias[0] += (strikes[0] - strikes[1]) / 2 - strikes[0]
        # i_bias = i_bias / ((strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1))
        # new_bias[dof:] = torch.nn.Parameter(i_bias) 
        # # print(new_bias[dof:])
        self.blackbox[-1].bias = torch.nn.Parameter(new_bias)
        self.scale = 1.0e8
        self.dof = dof
        self.num_knots = num_knots
        self.memo = None

    def forward(self, x, scale=None):
        # TODO control as correction to lstsq
        raw = self.blackbox(x)
        control = raw[:,:self.dof]
        knots = raw[:,self.dof:]
        knots = self.relu(knots)
        # print(control.T)
        if self.memo is None:
            flatx = x.reshape(-1)
            his = flatx[:(flatx.shape[0]-3)//3]
            lows = flatx[(flatx.shape[0]-3)//3:2*(flatx.shape[0] - 3)//3] # TODO Add Weighted least squares
            mids = (his + lows) / 2
            spreads = his-lows
            strikes = flatx[2*(flatx.shape[0] - 3)//3:-3]
            k_start = self.knot_skeleton.reshape(-1)[[0]]
            k_end = self.knot_skeleton.reshape(-1)[[-1]]
            init_basis, ini_sys = get_basis(self.base_system, self.knot_skeleton.reshape(-1)[1:-1], self.seeded_vecs, k_start, k_end)

            xstar = get_xstar(init_basis, self.knot_skeleton.reshape(-1)[1:-1], strikes,mids , ini_sys, spreads=spreads)
            self.memo=xstar
        else:
            xstar = self.memo
        knots = get_knots(self.strike_low.item(), self.strike_high.item(), knots) + self.knot_skeleton.reshape(knots.shape)
        print(knots.shape)
        if scale is None:
            scale = self.scale
        print(self.base_system.shape)
        k_start = self.knot_skeleton.reshape(-1)[[0]]
        k_end = self.knot_skeleton.reshape(-1)[[-1]]
        translate, sys = get_basis(self.base_system, knots[0,1:-1], self.seeded_vecs, k_start, k_end)
        control = control + xstar.reshape(1,-1)
        # polys = [np.polynomial.polynomial.Polynomial(list(reversed(sol[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((sol.shape[0])//4)]
        # plot_polys(polys, torch.hstack([torch.tensor(7), knots.reshape(-1), torch.tensor(9)]), 0, 'after')
        # ks = get_knot_system(self.base_system, knots, self.seeded_vecs)
        # raise Exception("Deliberate")
        # If we have take c' = (0,0,0,1,0,0,0,1,0,0,0,1...)
        # if x is a solution, A(x + c') also a solution
        # Hence Ac' = 0, hence Bc = c'
        # Hence We want to take Binv. B is tall, so we use torch.linalg.solve(B)
        # correction = crudePolyCorrect(translate @ control.T, knots, self.strike_low, self.strike_high)
        # correction = translate.T @ correction
        control1, l = removeArb(control.T, translate, knots, self.strike_low, self.strike_high, sys=sys)
        sol = translate @ control1
        sol = getBlockForm(sol)
        # sol = torch.hstack([differentiablePolyMul.apply(sol[i,:], sol[i,:], 23) for i in range(sol.shape[0])]).reshape(-1,1)

        return sol, knots 

    def train(self, data, strikes, epochs=1000, optimizer=None):
        scale=1.0
        lows, highs = data[0, (data.shape[1] - 3)//3:2 * (data.shape[1] - 3) // 3], data[0,:(data.shape[1] - 3)//3]
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1.0e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        S = data[0,-3]
        T = data[0,-2]
        R = data[0,-1]
        losses = []
        for e in range(epochs):
            print('Epoch:',e)
            # Loss and prediction
            pred, boundaries = self.forward(data, scale)
            pred = pred * 100


            polys = [np.polynomial.polynomial.Polynomial(pred[i,:].detach().numpy()) for i in range((pred.shape[0]))]

            # plot_polys([poly.deriv() for poly in polys], self.boundaries, e, 'dit')
            # plot_polys([poly.deriv(2) for poly in polys], self.boundaries, e, 'd2it')
            boundaries = boundaries.reshape(-1)
            # boundaries2 = [boundaries[i].item() for i in range(boundaries.shape[0])]
            transformed = transform(pred)
        
            loss = get_observe_losses(pred, boundaries, strikes, lows, highs)
            losses += [loss.item()]


            atmmid = 0.5 * (data[0, ((data.shape[1] - 3) // 3) //2] + data[0, (data.shape[1] - 3) // 3 + ((data.shape[1] - 3) // 3) //2]).item()
            if True: #e%10==0:
                print('it',e,'plotted')
                plot_polys([(lambda x, p=poly: p(x) ** 2) for poly in polys], torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'it', extra_points=[(strikes, torch.sqrt(lows)), (strikes, torch.sqrt(highs))], title=str(loss.item()))
                plot_polys([poly.deriv() for poly in polys], torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'dit', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))
                plot_polys([poly.deriv(2) for poly in polys], torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'ddit', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))
                tpolys = [np.polynomial.polynomial.Polynomial(transformed.detach().numpy()[i,:])  for i in range(transformed.shape[0])]
            
                plot_polys(tpolys, torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'tit','t')

                # gpolys = [partial(g, t=tpolys[i], p=polys[i]) for i in range(len(tpolys))] 
                # rns = [partial(rn_tv, w=polys[i], g=gpolys[i]) for i in range(len(gpolys))]

                # print(len(gpolys), len(rns))
                # plot_polys(gpolys, torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'git','g')
                # plot_polys(rns, torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'rnm','rnm')
                # rns = [partial(rn_transform, ls=S.item(), t=gpolys[i], w=polys[i], r=R.item(), T=T.item()) for i in range(len(tpolys))]
                # plot_polys(rns, torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'rn','rn custom')

                print('tit',e,'plotted')

            # print('Loss:', loss)
            # print(torch.autograd.grad(loss, pred, retain_graph=True, allow_unused=True))

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        plt.plot(losses)
        plt.savefig('./loss.png')
def plot_poly_from_output(sol, boundaries):
    polys = [np.polynomial.polynomial.Polynomial(list(sol[:,0].detach().numpy()[4*i:4*i+4])) for i in range((sol.shape[0])//4)]
    plot_polys(polys, boundaries, 0, 'custom')

def getBlockForm(poly, deg=3):
    newPolys = poly.reshape((-1,deg+1))
    return newPolys

def s_fs(sol, knots):
    # Takes block sol, applies after squaring
    knots = knots.reshape(-1)
    a = differentiablePolyEval.apply(differentiablePolyDeriv.apply(sol[0,:]),knots[0])
    b = differentiablePolyEval.apply(sol[0,:],knots[0]) - knots[0] * a # ax + b = p(x), b = p(x) - ax
    z = torch.tensor([0.0])
    s0 = torch.hstack([b,a,z,z,z,z,z])
    a = differentiablePolyEval.apply(differentiablePolyDeriv.apply(sol[-1,:]),knots[-1])
    b = differentiablePolyEval.apply(sol[-1,:],knots[-1]) - knots[-1] * a # ax + b = p(x), b = p(x) - ax
    z = torch.tensor([0.0])
    s1 = torch.hstack([b,a,z,z,z,z,z])
    return torch.vstack([s0, sol, s1])


def g(x, t, p):
    return t(x) / (4 * p * p)(x)
def d_sub(lk, w):
    wsk= lk + 1 
    return -lk / np.sqrt(w(wsk)) - np.sqrt(w(wsk))/2

def rn_tv(k, w, g):
    # norm is not proven or anything

    k = k - 1.0
    # print(np.exp(-d_sub(k,w)**2 / 2))
    # print(w(k+1))
    return np.exp(-d_sub(k, w)**2 / 2) * g(k + 1) / (np.sqrt(2 * np.pi * w(k+1)))

def transform(sol, scale=1):
    # Assume sol is of form a0, b0, c0, d0, a1, b1, ...
    transforms = []
    sol = sol
    sol_degree = sol.shape[1]
    max_degree = sol_degree *2 + (sol_degree - 1) * 2
    for i in range(0, sol.shape[0]):
        # Condition is (1 - kw' / 2w)^2 - w'^2/4(1/w + 1/4) + w''/2 >= 0
        # 1/(4w^2) (w'^2k^2) - w'^2 / 4w - w'k/w - w'^2 / 16 + w''/2 + 1
        # Mapping w = a, w' = b, w'' = c, k = x
        # Multiplying by 4 * w**2
        # Target: -1/4 a^2 b^2 + 2 a^2 c + 4 a^2 - a b^2 - 4 a b x + b^2 x^2

        # Max degree is degree a^2 *  b^2 = degree(3) * degree(3) * degree(2) *degree(2) = degree(10)

        # max_degree = w.degree() * 2 + (w.deriv()).degree() * 2 # With squaring - degree 6 * 2 + degree 6 * 2 - 1 = degree 23
        preservingPolyMul = lambda x, y: differentiablePolyMul.apply(x, y, max_degree)
        preservingPolyAdd = lambda x, y: differentiablePolyAdd.apply(x, y, max_degree)
        preservingPolySub = lambda x, y: differentiablePolySub.apply(x, y, max_degree)

        # preservingTensorPoly = lambda x: convertTensorPoly.apply(x, max_degree)

        a = sol[i,:]
        b = differentiablePolyDeriv.apply(a)
        c = differentiablePolyDeriv.apply(b)

        a2 = preservingPolyMul(a, a)
        b2 = preservingPolyMul(b,b)
        k = torch.tensor([0,1]) # Account for translation
        k2 = preservingPolyMul(k,k)
        cond = - 1 * preservingPolyMul(a2,b2) / 4.0 # -1/4 a^2b^2
        # print(cond)
        cond = preservingPolyAdd(cond, 2 * preservingPolyMul(a2, c)) # + 2a^2c
        cond = preservingPolyAdd(cond, 4 * a2) # + 4a^2
        cond = preservingPolySub(cond, preservingPolyMul(a, b2)) # - ab^2
        cond = preservingPolySub(cond, 4 * preservingPolyMul(a, preservingPolyMul(b, k))) # - 4 * a b k
        cond = preservingPolyAdd(cond, preservingPolyMul(b2,k2)) # + b^2 x^2
        transforms += [cond]
    ret_transforms = torch.zeros(len(transforms), max_degree+1)
    for i in range(len(transforms)):
        ret_transforms[i, :transforms[i].shape[0]] = transforms[i]
    return ret_transforms * scale

class convertTensorPoly(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, coeffs):

        return np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class differentiablePolyAdd(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeff1, coeff2, preserve_degree=None):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        result = torch.tensor(np.polyadd(poly1, poly2)[0].coef)
        if preserve_degree is not None:
            ret = torch.zeros((preserve_degree + 1,))
            ret[:result.shape[0]] = result
        else:
            ret = result
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        # Double check
        return grad_output, grad_output, None

class differentiablePolySub(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeff1, coeff2, preserve_degree=None):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        result = torch.tensor(np.polysub(poly1, poly2)[0].coef)
        if preserve_degree is not None:
            ret = torch.zeros((preserve_degree + 1,))
            ret[:result.shape[0]] = result
        else:
            ret = result
        return ret


    @staticmethod
    def backward(ctx, grad_output):
        # Double check

        return grad_output, -grad_output, None


class differentiablePolyMul(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeff1, coeff2, preserve_degree):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        ctx.save_for_backward(coeff1, coeff2, torch.tensor(preserve_degree))
        result = torch.tensor(np.polymul(poly1, poly2)[0].coef)
        if preserve_degree is not None:
            ret = torch.zeros((preserve_degree + 1,))
            ret[:result.shape[0]] = result
        else:
            ret = result
        return ret
    @staticmethod
    def backward(ctx, grad_output):
        # Takes dL/dc, c are coefficients of P
        # Returns [dL/dca, dL/dcb], where ca, cb are the coefficients of p(x), q(x) s.t P(x) = p(x) q(x)
        coeff1, coeff2, preserve_degree = ctx.saved_tensors
        preserve_degree = preserve_degree.item()
        ret1 = torch.zeros(coeff1.shape)
        ret2 = torch.zeros(coeff2.shape)
        for i in range(coeff1.shape[0]):
            for j in range(min(preserve_degree - i, coeff2.shape[0])):
                ret1[i] += grad_output[i+j] * coeff2[j]

        for i in range(coeff2.shape[0]):
            for j in range(min(preserve_degree - i, coeff1.shape[0])):
                ret2[i] += grad_output[i+j] * coeff1[j]
        # print('In grad: ',torch.any(torch.isnan(ret1)), torch.any(torch.isnan(ret2)))

        return ret1, ret2, None


class differentiablePolyEval(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeffs, x):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        ctx.save_for_backward(coeffs, torch.tensor(x))
        poly = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())


        return torch.tensor(poly(x))
    @staticmethod
    def backward(ctx, grad_output):
        # Only implementing derivative for coefficients rn
        coeffs, x = ctx.saved_tensors
        return_grad = torch.empty(coeffs.shape[0])
        poly = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())
        xder = torch.tensor(poly.deriv()(x.item()))
        for i in range(coeffs.shape[0]):
            return_grad[i] = x.item() ** i
        return return_grad * grad_output, xder * grad_output

class differentiablePolyRoots(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, coeffs, x0, x1):
        tol = 1.0e-5
        poly = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())
        roots = poly.roots()
        roots = roots.real[abs(roots.imag) < tol]
        roots = roots[(roots >= x0) & (roots <= x1)]
        roots = np.sort(roots)
        roots = torch.tensor(roots)
        ctx.save_for_backward(coeffs, roots)
        return roots

    @staticmethod
    def backward(ctx, grad_out):
        coeffs, roots = ctx.saved_tensors
        ders = torch.tensor(roots.shape[0], coeffs.shape[0])
        der = np.polynomial.polynomial.Polynomial(coeffs.detach()).deriv()
        # Jacobian (dy/dxi ...)
        # droots / dcoeffi
        for i in roots.shape[0]:
            ders[:,i] = - (roots ** i) / der(roots)
        return ders.T @ grad_out, None, None

# class differentiablePolyInteg(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, coeffs, x0, x1):
#         poly = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())
#         return torch.tensor(poly.integ(x1.detach().numpy()) - poly.integ(x0.detach().numpy()))

def polyLossT(cond_coeffs, bounds, sl, sh):
    bounds = torch.hstack([torch.tensor(sl).reshape(1,1), bounds, torch.tensor(sh).reshape(1,1)])
    loss = torch.tensor(0, dtype=torch.double)
    polys = [cond_coeffs[i,:] for i in range(cond_coeffs.shape[0])]

    for poly, (x0, x1) in zip(polys, zip(bounds[0,:-1],bounds[0,1:])):

        x0 = x0.item()
        x1 = x1.item()
        deriv = differentiablePolyDeriv.apply(poly)
        roots = differentiablePolyRoots.apply(deriv, x0, x1)
        for point in ([x0] + list(roots) + [x1]):
            loss = torch.min(torch.min(loss, differentiablePolyEval.apply(poly, torch.tensor(point))))
    return -1 * loss


class differentiableNormCdf(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.tensor(norm.cdf(x.detach().numpy()))
    
    @staticmethod
    def backward(ctx, x_out):
        x = ctx.saved_tensors[0]
        return torch.tensor(norm.pdf(x.detach().numpy())) * x_out


# class polyCorrect(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, coeffs, knots):
#         sol = sol.reshape(-1)
#         bounds = knots.reshape(-1,1)
#         for i in range(0, sol.shape[0], 4):
#             wd = (torch.flip(sol[i:(i+4)],(0,)))
#             wd 
#             x0 = knots[i // 4]
#             x1 = knots[i // 4 + 1]
#             roots = wd.roots()
#             roots = roots[(roots >= x0) & (roots <= x1)]

# class polyRoots(torch.autograd.Function):

#     def forward(ctx, coeffs):
#         tol = 1.0e-5
#         w = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())
#         r = w.roots()
#         r = r.real[abs(r.imag) < tol]
#         ctx.save_for_backward(torch.tensor(r), coeffs)
#         return torch.tensor(r)
    
#     def backward(ctx, grad_output):
#         return 
        
class differentiablePolyDeriv(torch.autograd.Function):

    def forward(ctx, coeffs):
        w = convertTensorPoly.apply(coeffs)
        ctx.save_for_backward(coeffs)
        return torch.tensor(w.deriv().coef)
    
    def backward(ctx, grad_output):
        coeffs = ctx.saved_tensors[0]
        ret = torch.arange(coeffs.shape[0])
        ret[1:] = ret[1:] * grad_output
        return ret 

def crudePolyCorrect(ocoeffs, boundaries, sl, sh):
    lr = 1.0
    tol=1.0e-6
    max_epochs = 100
    scale=1
    # Want to apply normalisation s.t x_min k + a = 0.9
    #                                 x_max k + a = 1.1
    # Then we get k = 0.2 / (x_min - x_max), a = 0.9
    # print('arb bounds')
    # print(boundaries)
    # Normalize data:
    
    polys = ocoeffs.reshape((-1, 4))
    ppolys = [np.polynomial.polynomial.Polynomial(polys.detach().numpy()[i,:])  for i in range(polys.shape[0])]
    plot_polys(ppolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), -2, 'pint','v')        

    loss = scale * polyLossT(polys, boundaries, sl, sh) # + scale * polyLoss.apply(sol, boundaries, sl, sh)
    print('Neg pre', loss)

    addition = torch.ones((polys.shape[0], 1)) * (loss.item() + 1.0e-9)
    zeros = torch.zeros(polys.shape[0], 3)
    addition = torch.hstack([addition, zeros])
    polys = polys + addition
    ppolys = [np.polynomial.polynomial.Polynomial(polys.detach().numpy()[i,:])  for i in range(polys.shape[0])]
    plot_polys(ppolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), -1, 'pint','v')        


    polys = polys.reshape(-1,1)

    return polys

class polyLoss(torch.autograd.Function):
      
    @staticmethod
    def forward(ctx, cond_coeffs, bounds, sl, sh):
        tol = 1.0e-5 # Tolerance for discarding imaginary roots
        bounds = torch.hstack([torch.tensor(sl).reshape(1,1), bounds, torch.tensor(sh).reshape(1,1)])
        # 1. Convert into polynomials from output:
        polys = [np.polynomial.polynomial.Polynomial(cond_coeffs[i,:]) for i in range(cond_coeffs.shape[0])]

        # Polys is our list of piecewise condition polynomials
        coeff_ders = torch.zeros(cond_coeffs.shape)
        bounds_ders = torch.zeros(bounds.shape)
        i = 0
        all_roots = 0
        loss = torch.tensor(0, dtype=torch.double)
        # print(bounds)
        # print(polys)
        for poly, (x0, x1) in zip(polys, zip(bounds[0,:-1],bounds[0,1:])):

            # print(poly.coef)
            # print(poly)
            x0 = x0.item()
            x1 = x1.item()
            polyder = poly.deriv()
            roots = poly.roots()
            roots = roots.real[abs(roots.imag) < tol]
            # print(roots,x0, x1)
            # print(poly(x0),poly(x1))
            roots = roots[(roots >= x0) & (roots <= x1)]
            roots = np.sort(roots)
            # print(roots)
            for lo, hi in zip([x0] + list(roots), list(roots) + [x1]):

                integral = poly.integ()
                loss += min(0, integral(hi) - integral(lo) - 1.0e-5)
                # print(lo, hi, poly(lo), poly(hi))
                # print(min(0, integral(hi) - integral(lo)), roots.shape)
                for j in range(coeff_ders.shape[1]):
                    # spline i, coefficient j
                    if (polyder(lo) != 0) and (polyder(hi) != 0):
                        coeff_ders[i,j] += (-1 * poly(hi) * hi ** j / polyder(hi)) + (poly(lo) * lo ** j / polyder(lo))
            i += 1
        ctx.save_for_backward(coeff_ders, bounds_ders)
        return -1 * loss

    @staticmethod
    def backward(ctx, grad_output):
        # Add gradient support for boundaries
        coeff_ders, bounds_ders = ctx.saved_tensors
        # print(coeff_ders.mean())
        return -coeff_ders, None, None, None


#  def rn_from_splines_transforms(splines, transforms):
#     ret_funcs = []
#     dsub = -sin,
#     for i in len(splines):
#         ret += [lambda k, =]

def plot_polys(polys, boundaries, num, name, title='', extra_points=None, labels=False):
    # print("./"+name+str(num)+".png")
    boundaries = boundaries.detach()
    boundary_spaces = [torch.tensor(np.linspace(x,y,num=100)) for x, y in zip(boundaries[:-1],boundaries[1:])]
    plot_points = np.hstack([poly(space) for poly, space in zip(polys, boundary_spaces)])
    # print('in plot')
    # for i in range(len(boundary_spaces)):
    #     print(polys[i])
    #     print(boundary_spaces[i][0], boundary_spaces[i][-1], polys[i](boundary_spaces[i][0]),polys[i](boundary_spaces[i][-1]))
    #     print('max', np.log(max([boundary_spaces[i][-1] ** j * polys[i].coef[j] for j in range(polys[i].coef.shape[0])])))
    points = np.hstack([i.detach() for i in boundary_spaces])
    plt.figure(figsize=(32,32))
    plt.plot(points.reshape(-1), plot_points.reshape(-1))

    if extra_points is not None:
        if not labels:
            for x, y in extra_points:
                plt.scatter(x, y)
        else:
            for x, y, z in extra_points:
                plt.scatter(x,y,label=z)
            plt.legend(loc='upper right')
    plt.vlines(boundaries, ymin = min(plot_points.reshape(-1)), ymax = max(plot_points.reshape(-1)), color = 'red')
    plt.hlines([0.0], xmin=boundaries[0], xmax=boundaries[-1], color='black')
    plt.suptitle(title)
    plt.savefig("./"+name+str(num)+".png")
    plt.close()
    # for i in polys:
    #     print(i)
    # for j in boundaries:
    #     print(j)
    # raise Exception('Deliberate')

def full_loss(coeffs, boundaries, strikes, lows, highs, scale=10000000):
    transformed = transform(coeffs, scale=scale)
    loss = (scale * polyLoss.apply(transformed, boundaries)).sqrt()
    print("Arb loss:", loss)
    if loss.item() > -1.0e-5:
        # print('yes!')
        loss = loss + get_observe_losses(coeffs, boundaries, strikes, lows, highs)
    return loss

def get_observe_losses(coeffs, boundaries, strikes, lows, highs):
    # Pred: boundaries, strikes sorted
    # print("Testing:: ")
    # polys_test = [np.polynomial.polynomial.Polynomial(list(reversed(coeffs[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((coeffs.shape[0])//4)]
    print(coeffs.shape)
    polys = [lambda x, p=coeffs[i,:]: (differentiablePolyEval.apply(p, x))**2 for i in range((coeffs.shape[0]))]
    last_bound = 0
    loss = torch.tensor(0.0).double()
    zero = torch.tensor(0.0).double()
    flag = False
    flag2 = False
    errs = []
    for strike, low, high in zip(strikes, lows, highs):
        while last_bound < len(boundaries) and boundaries[last_bound] < strike:
            last_bound += 1
        predicted = polys[last_bound](strike)
        if torch.any(predicted < 0) or flag:
            flag = True
            if not flag:
                loss = torch.tensor(0.0).double()
            loss = loss + torch.abs(min(zero, predicted))
            print(predicted, strike)
            print(last_bound, polys[last_bound])
        # print(predicted, strike)
        mid = (high + low) / 2.0
        out = (high - low) / 2.0 # how far is bid/ask
        err = (predicted - mid).abs()
        if torch.max(err - out, zero).item() > 0:
            flag2 = True
            errs += [(strike, torch.max(err - out, zero).item())]
        bl = torch.max(err - out, zero)
        loss = loss + (bl + bl**2  + torch.max(out - err, zero)  * err**2 / strikes.reshape(-1,1).shape[0]) / out
    if flag:
        print('Neg loss', loss.item())
    elif flag2:
        print('Bound Loss', loss.item())
        print(errs)
    else:
        print('Mid loss', loss.item())
    
    return loss / len(strikes)

def removeArb(control, translate, boundaries, sl, sh, sys=None):
    lr = 1
    gamma = 0.9
    tol=1.0e-6
    max_epochs = 100
    scale=1
    # Want to apply normalisation s.t x_min k + a = 0.9
    #                                 x_max k + a = 1.1
    # Then we get k = 0.2 / (x_min - x_max), a = 0.9
    # Normalize data:
    for e in range(max_epochs):
        # print(control)
        sol0 = translate @ control
        sol0 = getBlockForm(sol0)
        print(sol0)
        polys = [np.polynomial.polynomial.Polynomial(sol0.detach().numpy()[i,:])  for i in range(sol0.shape[0])]
        plot_polys(polys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1)[1:-1], torch.tensor(sh)]), e, 'pint','p')        
        plot_polys([poly.deriv(1) for poly in polys], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1)[1:-1], torch.tensor(sh)]), e, 'pint1_','p')        
        plot_polys([poly.deriv(2) for poly in polys], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1)[1:-1], torch.tensor(sh)]), e, 'pint2_','p')        

        sol = [differentiablePolyMul.apply(sol0[i,:], sol0[i,:], 6).reshape(1,-1) for i in range(sol0.shape[0])]
        sol = torch.vstack(sol)
        sol = s_fs(sol, boundaries)
        # raise Exception("D")
        transformed = transform(sol)
        # plot_polys(tpolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'tint','t')        
        tpolys = [np.polynomial.polynomial.Polynomial(transformed.detach().numpy()[i,:])  for i in range(transformed.shape[0])]
        plot_polys(tpolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'tint','t')        
        polys = [np.polynomial.polynomial.Polynomial(sol0.detach().numpy()[i,:])  for i in range(sol0.shape[0])]
        polys2 = [np.polynomial.polynomial.Polynomial(sol.detach().numpy()[i,:])  for i in range(sol.shape[0])]


        plot_polys(polys2, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'pint_2_','p')        
        plot_polys([poly.deriv(1) for poly in polys2], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'pint1_2_','p')        
        plot_polys([poly.deriv(2) for poly in polys2], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'pint2_2_','p')        


        loss = polyLoss.apply(transformed, boundaries, sl, sh) # + scale * polyLoss.apply(sol, boundaries, sl, sh)
        ##########################
        print('Arb loss', loss, e)
        # loss = loss + scale * polyLossT(polys, boundaries, sl, sh) # + scale * polyLoss.apply(sol, boundaries, sl, sh)
        print('Total loss',loss)
        ##########################
        # ppolys = [np.polynomial.polynomial.Polynomial(polys.detach().numpy()[i,:])  for i in range(polys.shape[0])]
        # plot_polys(ppolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'pint','v')        

        if loss < tol:
            break
        grad = torch.autograd.grad(loss, control, retain_graph=True)[0]
        print(torch.autograd.grad(loss, sol0, retain_graph=True))
        control = control - lr * grad   # TODO differentiable_update
        lr = lr * gamma
    # print(translate @ grad)

    print("With remaining arb:", loss.item(), 'over', e, 'epochs')

    return control, loss # Returns control that satisfies arbitrage condition


def getJacobian(Y, X):
    print(Y)
    print(X)
    Z = torch.empty((Y.shape[0], X.shape[0]))
    for i in range(X.shape[0] - 1):
        Z[:,i] = torch.autograd.grad(Y[i],X, retain_graph=True)[0][:,0]
    Z[:, Y.shape[0] - 1] = torch.autograd.grad(Y[Y.shape[0] -1], X, retain_graph=False)[0][:,0]
    return Z

#  def BSC_from_w(w,k,r):
#     # w(k) = sigma^2t
#     # k = ln(K / St) + 1
#     d1 = 1/(k-1) + (r +)