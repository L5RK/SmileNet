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
    system = torch.zeros(((num_knots * 3) + 4, (4 * (num_knots + 1)))).double()

    for i in range(0, num_knots, 1):
        # For every knot, enforce c0 continuity on the two adjoining splines
        system[3 * i,   (4*i):(4*i)+8] = torch.tensor([1,1,1,1,-1,-1,-1,-1]) # C0
        system[3 * i+1, (4* i):(4*i)+8] = torch.tensor([3,2,1,0,-3,-2,-1,0]) # C1
        system[3 * i+2, (4* i):(4*i)+8] = torch.tensor([6,2,0,0,-6,-2,0,0]) # C2
    system[-1,0] = 1
    system[-2,1] = 1
    system[-3,-3] = 1
    system[-4,-4] = 1
    # system.requires_grad_()

    return system

def get_basis(system, knots, seeded_vecs):
    p1 = torch.tensor([3,2,1,0])
    p2 = torch.tensor([2,1,0,0])
    p3 = torch.tensor([1,0,0,0])
    knots = knots.reshape(-1,1)
    kp1 = (knots ** p1).repeat(1,knots.shape[0] + 1)
    kp2 = (knots ** p2).repeat(1,knots.shape[0] + 1)
    kp3 = (knots ** p3).repeat(1,knots.shape[0] + 1)
    knot_powers = torch.hstack([kp1,kp2,kp3]).reshape(-1, (knots.shape[0] + 1) * 4)
    new_sys = knot_powers * system[:-4,:]
    new_sys = torch.vstack([new_sys, system[-4:,:]])
    Q, R = torch.linalg.qr(new_sys.T)
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
    # print("in basis")
    # print(basis.T @ basis)
    return basis, new_sys

def get_vector_seed(num_knots):
    dof = num_knots + 4 - 4 # Subject to change -> Linear wings -4 for linear wings
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

def get_xstar(basis, knots, strikes, mids,ini_sys):
    # Pred - knot n is before strike n
    # Pred - strikes are sorted increasing
    # pred - knots are sorted increasing
    eqs = torch.zeros(strikes.shape[0], 4 * (knots.shape[0] + 1)).double()
    inds_offset = torch.arange(knots.shape[0] - 1)
    knotsR = knots.repeat((strikes.shape[0],1)).T
    inds = torch.where((knotsR[inds_offset] < strikes) & (knotsR[inds_offset+1] > strikes))[0]
    inds = torch.hstack([torch.tensor([0]), inds, torch.tensor(knots.shape[0] -1)])
    powers = torch.tensor([3,2,1,0])
    strike_powers = (strikes.reshape(-1,1) ** powers).repeat(1, knots.shape[0] + 1) # number polys = number knots + 1
    for i in range(inds.shape[0]):
        eqs[i,4*inds[i]:4*(inds[i]+1)] = 1

    eqs = strike_powers * eqs
    # print(knots)
    # print('eqs beg')
    # for i in range(eqs.shape[0]):
    #     print(eqs[i,:])
    # print('eqs done')

    # print(strikes)
    # print(mids)
    # print(knots)
    bsol  = torch.linalg.lstsq(eqs @ basis, mids.reshape(-1,1))[0]
    sol = basis @ bsol
    # print(sol)
    polys = [np.polynomial.polynomial.Polynomial(list(reversed(sol[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((sol.shape[0])//4)]
    plot_polys(polys, torch.hstack([torch.tensor(6), knots, torch.tensor(10)]), 0, 'before', extra_points=zip(strikes, mids))
    plot_poly_from_output(sol, knots)
    # print(bsol)
    return bsol
    # strike_powers (x0 ** 3, x0)

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
        dof = num_knots + 4 - 4 # Linear wings -> -4 degrees of freedom
        strike_knots = []
        count = 0
        # initial_knots = (torch.arange(num_knots)) * (strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1)+ (strikes[0] + strikes[1]) / 2
        strikes = torch.exp(strikes)
        initial_knots = (torch.arange(num_knots)) * ((strikes[-10] + strikes[-1])/2 - (strikes[10] + strikes[0]) / 2) / (num_knots - 1)+ (strikes[0] + strikes[10]) / 2
        strikes = torch.log(strikes)
        initial_knots = torch.log(initial_knots)
        # initial_knots = strikes[[i * strikes.shape[0] // (num_knots+2) for i in range(1,num_knots+1)]]
        self.strike_low = strikes[0]
        self.strike_high = strikes[-1]
        self.seeded_vecs = get_vector_seed(num_knots)
        self.base_system = get_system_matrix(num_knots)
        init_basis, ini_sys = get_basis(self.base_system, initial_knots, self.seeded_vecs)
        xstar = get_xstar(init_basis, initial_knots, strikes, mids, ini_sys)

        strike_num = strikes.shape[0]
        print(strike_num * 3)
        self.blackbox = torch.nn.Sequential(
            torch.nn.Linear(strike_num * 3 + 3, strike_num * 4 + 3), # 2*strikes for bid, ask + 1*strikes for strike pos + 3 for logS, T, R
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4 + 3, strike_num * 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 8, strike_num * 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 16, dof + num_knots) #- len(strike_knots)),
        )
        self.relu = torch.nn.ReLU()
        self.knot_skeleton = initial_knots
        print(initial_knots)
        print(strikes[-1])
        for i in self.blackbox:
            if not isinstance(i, torch.nn.LeakyReLU):
                i.weight = torch.nn.Parameter(torch.zeros(i.weight.shape).double() / 100)
                i.bias = torch.nn.Parameter(torch.zeros(i.bias.shape).double() / 100)
        new_bias = torch.zeros(self.blackbox[-1].bias.shape)
        new_bias[:dof] = xstar[:,0]
        # i_bias = torch.ones(num_knots).double() 
        # i_bias[0] += (strikes[0] - strikes[1]) / 2 - strikes[0]
        # i_bias = i_bias / ((strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1))
        # new_bias[dof:] = torch.nn.Parameter(i_bias) 
        # # print(new_bias[dof:])
        self.blackbox[-1].bias = torch.nn.Parameter(new_bias)
        self.scale = 1.0e8
        self.dof = dof
        self.num_knots = num_knots
        self.epsilon = 0.1

    def forward(self, x, scale=None):
        raw = self.blackbox(x)
        control = raw[:,:self.dof]
        knots = raw[:,self.dof:]
        knots = self.relu(knots)
        print('control')
        print(control.T)
        knots = get_knots(self.strike_low.item(), self.strike_high.item(), knots) + self.knot_skeleton.reshape(knots.shape) 
        print(knots)

        if scale is None:
            scale = self.scale

        translate, sys = get_basis(self.base_system, knots, self.seeded_vecs)
        sol = translate @ control.T
        # print(sol)
        polys = [np.polynomial.polynomial.Polynomial(list(reversed(sol[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((sol.shape[0])//4)]
        plot_polys(polys, torch.hstack([torch.tensor(7), knots.reshape(-1), torch.tensor(9)]), 0, 'after')
        # ks = get_knot_system(self.base_system, knots, self.seeded_vecs)
        # print(ks.shape)
        # print(translate.shape, control.shape)
        # print( ks @ (translate @ control.T))
        # raise Exception("Deliberate")
        # If we have take c' = (0,0,0,1,0,0,0,1,0,0,0,1...)
        # if x is a solution, A(x + c') also a solution
        # Hence Ac' = 0, hence Bc = c'
        # Hence We want to take Binv. B is tall, so we use torch.linalg.solve(B)
        correction = crudePolyCorrect(translate @ control.T, knots, self.strike_low, self.strike_high)
        # correction = translate @ correction
        correction = translate.T @ correction

        sol = remove_arb(correction, translate, knots, scale=scale)
        # print(knot_powers @ sol)
        # print(knots)
        # print("translation")
        # assert(translate.T @ knot_powers.T) # Each of the columns should lie in null spaces of knot_powers.T
        #                                  # Hence each of the columns 
        # knots = torch.hstack([torch.tensor(self.strike_low).reshape(1,1), knots, torch.tensor(self.strike_high).reshape(1,1)])
        # print(sol)
        # raise Exception("Deliberate")

        return sol, knots 

    def train(self, data, strikes, epochs=1000, optimizer=None):
        scale=1.0e5
        lows, highs = data[0, data.shape[1]//3:2 * data.shape[1] // 3], data[0,:data.shape[1]//3]
        if optimizer is None:
            optimizer = torch.optim.Adamax(self.parameters(), lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        for e in range(epochs):
            print('Epoch:',e)
            # Loss and prediction
            pred, boundaries = self.forward(data, scale)
            # print(pred)
            polys = [np.polynomial.polynomial.Polynomial(torch.flip((pred[:,0][4*i:4*i+4]), dims=(0,)).detach().numpy()) for i in range((pred.shape[0])//4)]

            # plot_polys([poly.deriv() for poly in polys], self.boundaries, e, 'dit')
            # plot_polys([poly.deriv(2) for poly in polys], self.boundaries, e, 'd2it')
            boundaries = boundaries.reshape(-1)
            # boundaries2 = [boundaries[i].item() for i in range(boundaries.shape[0])]
            transformed = transform(pred, scale=scale)

            loss = get_observe_losses(pred, boundaries, strikes, lows, highs)
            # print(torch.autograd.grad(loss, pred, retain_graph=True, allow_unused=True))
            # print(torch.autograd.grad(loss, boundaries))
            if e%10==0:
                print('it',e,'plotted')
                print(boundaries)
                # print(torch.autograd.grad(loss, boundaries, retain_graph=True))
                plot_polys(polys, torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'it', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))
                plot_polys([poly.deriv() for poly in polys], torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'dit', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))
                plot_polys([poly.deriv(2) for poly in polys], torch.hstack([torch.tensor(self.strike_low) , boundaries, torch.tensor(self.strike_high)]), e, 'ddit', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))

                polys = [np.polynomial.polynomial.Polynomial(transformed.detach().numpy()[i,:]) for i in range(transformed.shape[0])]

                plot_polys(polys, boundaries, e, 'tit',str(loss.item()))
                print('tit',e,'plotted')

            print('Loss:', loss)
            # print(torch.autograd.grad(loss, pred, retain_graph=True, allow_unused=True))
            if e == epochs - 1:
                print(pred)
            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

def plot_poly_from_output(sol, boundaries):
    # print(boundaries)
    polys = [np.polynomial.polynomial.Polynomial(list(reversed(sol[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((sol.shape[0])//4)]
    plot_polys(polys, boundaries, 0, 'custom')


def transform(sol, scale):
    # Assume sol is of form a0, b0, c0, d0, a1, b1, ...
    transforms = []
    sol = sol[:,0] / scale
    for i in range(0, sol.shape[0], 4):
        # Condition is (1 - kw' / 2w)^2 - w'^2/4(1/w + 1/4) + w''/2 >= 0
        # 1/(4w^2) (w'^2k^2) - w'^2 / 4w - w'k/w - w'^2 / 16 + w''/2 + 1
        # Mapping w = a, w' = b, w'' = c, k = x
        # Multiplying by 4 * w**2
        # Target: -1/4 a^2 b^2 + 2 a^2 c + 4 a^2 - a b^2 - 4 a b x + b^2 x^2

        # Max degree is degree a^2 *  b^2 = degree(3) * degree(3) * degree(2) *degree(2) = degree(10)

        w = convertTensorPoly.apply(torch.flip(sol[i:(i+4)],(0,)))
        # max_degree = w.degree() * 2 + (w.deriv()).degree() * 2
        max_degree = 10
        preservingPolyMul = lambda x, y: differentiablePolyMul.apply(x, y, max_degree)
        # preservingTensorPoly = lambda x: convertTensorPoly.apply(x, max_degree)

        a = torch.flip(sol[i:(i+4)], dims=(0,))
        b = torch.tensor(w.deriv().coef)
        c = torch.tensor(w.deriv(2).coef)

        a2 = preservingPolyMul(a, a)
        b2 = preservingPolyMul(b,b)
        k = torch.tensor([0,1])
        k2 = preservingPolyMul(k,k)
        cond = - 1 * preservingPolyMul(a2,b2) / 4.0 # -1/4 a^2b^2
        cond = differentiablePolyAdd.apply(cond, 2 * preservingPolyMul(a2, c)) # + 2a^2c
        cond = differentiablePolyAdd.apply(cond, 4 * a2) # + 4a^2
        cond = differentiablePolySub.apply(cond, preservingPolyMul(a, b2)) # - ab^2
        cond = differentiablePolySub.apply(cond, 4 * preservingPolyMul(a, preservingPolyMul(b, k))) # - 4 * a b k
        cond = differentiablePolyAdd.apply(cond, preservingPolyMul(b2,k2)) # + b^2 x^2
        print(np.polynomial.polynomial.Polynomial(cond.detach().numpy()))
        transforms += [cond]
    ret_transforms = torch.zeros(len(transforms), max(map(lambda x: x.shape[0], transforms)))
    for i in range(len(transforms)):
        ret_transforms[i, :transforms[i].shape[0]] = transforms[i]
    return ret_transforms

class convertTensorPoly(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, coeffs):

        return np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class differentiablePolyAdd(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeff1, coeff2):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        return torch.tensor(np.polyadd(poly1, poly2)[0].coef)

    def backward(ctx, grad_output):
        # Double check
        return grad_output, grad_output, None

class differentiablePolySub(torch.autograd.Function):
     
    @staticmethod
    def forward(ctx, coeff1, coeff2):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        return torch.tensor(np.polysub(poly1, poly2)[0].coef)

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
        # print(poly)
        # print('Inside', poly(x))
        return torch.tensor(poly(x))

    def backward(ctx, grad_output):
        # Only implementing derivative for coefficients rn
        coeffs, x = ctx.saved_tensors
        return_grad = torch.empty(coeffs.shape[0])
        poly = np.polynomial.polynomial.Polynomial(coeffs.detach().numpy())
        xder = torch.tensor(poly.deriv()(x.item()))
        for i in range(coeffs.shape[0]):
            return_grad[i] = x.item() ** i
        return return_grad * grad_output, xder * grad_output

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
        
# class differentiablePolyDeriv(torch.autograd.Function):

#     def forward(ctx, coeffs):
#         w = convertTensorPoly.apply(coeffs)
#         return torch.tensor(w.deriv().coef)
    
#     def backward(ctx, grad_output):
#         return torch.arange(grad_output.shape) * grad_output

def crudePolyCorrect(ocoeffs, boundaries, sl, sh):
    # print(ocoeffs)
    coeffs = ocoeffs.reshape(-1)
    boundaries = boundaries.reshape(-1)
    polys = [lambda x, p=coeffs[4*i:4*i+4].flip(dims=(0,)): differentiablePolyEval.apply(p, x) for i in range((coeffs.shape[0])//4)]
    minimum = torch.tensor(0.0)
    minimum.requires_grad_()
    for i in range(len(polys)):
        # print(coeffs[4*i:4*i+4])
        # print(coeffs[4*i:4*i+4].flip(dims=(0,)))
        if i != 0:
            minimum = torch.min(polys[i](boundaries[i-1]), minimum)
        else:
            minimum = torch.min(polys[i](sl), minimum)
        if i != len(polys) - 1:
            minimum = torch.min(polys[i](boundaries[i]),minimum)
        else:
            minimum = torch.min(polys[i](sh), minimum)
        # print(polys[i](boundaries[i]))
        # print(boundaries[i])
        # print(minimum)
    # print(torch.autograd.grad(minimum, coeffs, retain_graph=True))
    mask = ((torch.arange(coeffs.shape[0]) + 1) % 4 == 0)
    coeffs[mask] = coeffs[mask] - minimum
    ret = coeffs.reshape(-1,1)
    # print(torch.autograd.grad(ret[3,0],ocoeffs, retain_graph=True))
    # print(torch.autograd.grad(ret[3,0], minimum, retain_graph=True, allow_unused=True))
    # print([torch.autograd.grad(ret[i,0], ocoeffs, retain_graph=True) for i in range(ret.shape[0])])
    return ret

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
        for poly, (x0, x1) in zip(polys, zip(bounds[:-1],bounds[1:])):
            # print("------------")
            # print(poly.coef)
            polyder = poly.deriv()
            roots = poly.roots()
            roots = roots.real[abs(roots.imag) < tol]
            roots = roots[(roots >= x0) & (roots <= x1)]
            all_roots += roots.shape[0]
            roots = np.sort(roots)
            for lo, hi in zip([x0] + list(roots), list(roots) + [x1]):

                integral = poly.integ()
                loss += min(0, integral(hi) - integral(lo))
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
        return coeff_ders, None

        
def plot_polys(polys, boundaries, num, name, title='', extra_points=None):
    # print("./"+name+str(num)+".png")
    boundaries = boundaries.detach()
    boundary_spaces = [np.linspace(x,y,num=1000) for x, y in zip(boundaries[:-1],boundaries[1:])]
    plot_points = np.hstack([poly(space) for poly, space in zip(polys, boundary_spaces)])
    # print('in plot')
    # for i in range(len(boundary_spaces)):
    #     print(polys[i])
    #     print(boundary_spaces[i][0], boundary_spaces[i][-1], polys[i](boundary_spaces[i][0]),polys[i](boundary_spaces[i][-1]))
    #     print('max', np.log(max([boundary_spaces[i][-1] ** j * polys[i].coef[j] for j in range(polys[i].coef.shape[0])])))
    points = np.hstack(boundary_spaces)
    plt.plot(points, plot_points)
    if extra_points is not None:
        for x, y in extra_points:
            plt.scatter(x, y)
    plt.vlines(boundaries, ymin = min(plot_points), ymax = max(plot_points), color = 'red')
    plt.title(title)
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
    polys = [lambda x, p=coeffs[:,0][4*i:4*i+4].flip(dims=(0,)): differentiablePolyEval.apply(p, x) for i in range((coeffs.shape[0])//4)]
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
        if torch.max(err - out, zero).item() > 1.0e-5:
            flag2 = True
            errs += [(strike, torch.max(err - out, zero).item())]
        loss = loss + torch.max(err - out, zero)**2  + err / 10
    if flag:
        print('Neg loss', loss.item())
    elif flag2:
        print('Bound Loss', loss.item())
        print(errs)
    else:
        print('Mid loss', loss.item())
    
    return loss / len(strikes)

def remove_arb(control, translate, boundaries, scale=1.0e7, tol=1.0e-7):

    lr = 1.0e-5
    max_epochs = 100
    # Want to apply normalisation s.t x_min k + a = 0.9
    #                                 x_max k + a = 1.1
    # Then we get k = 0.2 / (x_min - x_max), a = 0.9
    
    # Normalize data:
    for e in range(max_epochs):
        sol = translate @ control
        # polys = [np.polynomial.polynomial.Polynomial(list(reversed(pred[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((pred.shape[0])//4)]
        transformed = transform(sol, a, b, scale=scale)
        loss = polyLoss.apply(transformed, boundaries, 1-epsilon, 1+epsilon) + polyLoss.apply(sol, boundaries, 1-epsilon, 1+epsilon)
        if loss < tol:
            break
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        grad = torch.autograd.grad(loss, control, retain_graph=True)[0]
        control = control - lr * grad
    print("With remaining arb:", loss.item())
    sol = translate @ control
    return sol # Returns control that satisfies arbitrage condition