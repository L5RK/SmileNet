import torch
import numpy as np
from scipy.stats import norm
from functools import partial
import matplotlib.pyplot as plt

# Neural Net:
# Very simple rn - getting ideas down

torch.set_default_dtype(torch.double)

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
    print('System shape:', system.shape, 'With', len(knots),'knots')
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
    for i, x in strike_knots[:-1]: # strike_knots are knots that correspond to a known strike price. Since C0 continuity already enforced, We only enforce one equality per knot on midprice
        # print(i, x)
        new_arr = np.zeros(X.shape[1])
        new_arr[4 * i] = x ** 3
        new_arr[4 * i + 1] = x ** 2
        new_arr[4 * i + 2] = x
        new_arr[4 * i + 3] = 1
        constraints += [new_arr]
        # print(new_arr, i,x)        
    # print('Xshape', X.shape)
    new_arr = np.zeros(X.shape[1])
    i, x = strike_knots[-1]
    i = i - 1 # We now are taking the right constraint, before were taking left
    new_arr[4 * i] = x ** 3
    new_arr[4 * i + 1] = x ** 2
    new_arr[4 * i + 2] = x
    new_arr[4 * i + 3] = 1
    constraints += [new_arr]

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

    def __init__(self, boundaries, strikes, mids):
        super().__init__()
        # Blackbox layers

        dof = len(boundaries) + 4 - 2 # Last 2 boundaries do not impact dof
        strike_knots = []
        count = 0
        # for i in range(len(boundaries)):
        #     if boundaries[i] == strikes[count]:
        #         strike_knots += [(i,boundaries[count])]
        #         count += 1
        for strike in strikes:
            while count < len(boundaries) and boundaries[count] <= strike:
                if boundaries[count] == strike:
                    strike_knots += [(count, boundaries[count])]
                count += 1
        print('Strike knots shape', len(strike_knots))
        basis, X = get_translation_matrix(boundaries[1:-1], dof)
        _, xstar,xdash = mid_constrained(strike_knots, mids, dof, X)
        self.boundaries = boundaries
        self.system = X
        print(basis.shape, X.shape, xstar.shape)
        strike_num = strikes.shape[0]
        self.blackbox = torch.nn.Sequential(
            torch.nn.Linear(strike_num * 2, strike_num * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4, strike_num * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4, dof) #- len(strike_knots)),
        )

        # self.blackbox.apply(lambda m: m.weight.data.fill_(0.0) if isinstance(m, torch.nn.Linear) else None)
        # self.blackbox.apply(lambda m: m.bias.data.fill_(0.0)  if isinstance(m, torch.nn.Linear) else None)
        # print(xstar.shape, 'xstar')
        # self.blackbox[-1].bias.data = torch.tensor(-1 * xstar.T).float()
        # Assuming stack returns [lambda_1, lambda_2, ... lambda_dof], shape(1 ,dof)
        # basis of shape (4(m+1),dof)
        self.xstar = xstar
        for i in self.blackbox:
            if not isinstance(i, torch.nn.LeakyReLU):
                i.weight = torch.nn.Parameter(torch.zeros(i.weight.shape).double() / 100)
                i.bias = torch.nn.Parameter(torch.zeros(i.bias.shape).double() / 100)
        print(self.xstar.shape)
        print(basis.shape)
        print((basis.T @ self.xstar).shape)
        print(self.blackbox[-1].bias.shape)
        self.blackbox[-1].bias = torch.nn.Parameter(torch.tensor(basis.T @ self.xstar).reshape(-1).double())
        self.translate = torch.tensor(basis.T).double() # we get coefficients vector from v @ w.T (assuming w itself is a row vector)



    def forward(self, x):
        control = self.blackbox(x)
        sol = self.translate.T @ control.T # Outputs coefficients
        # if not sanitycheck else sol, (torch.tensor(self.system) @ sol).reshape(-1,1)[int(-x.shape[1] / 2) :]
        # print('Sol shape', sol.shape)
        return sol 

    def train(self, data, strikes, epochs=100, optimizer=None):

        scale = 1.0e7
        lows, highs =  data[0, data.shape[1]//2:], data[0,:data.shape[1]//2]
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-06)
        for e in range(epochs):
            print('Epoch:',e)
            # Loss and prediction
            pred = self.forward(data)
            polys = [np.polynomial.polynomial.Polynomial(list(reversed(pred[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((pred.shape[0])//4)]

            # plot_polys([poly.deriv() for poly in polys], self.boundaries, e, 'dit')
            # plot_polys([poly.deriv(2) for poly in polys], self.boundaries, e, 'd2it')

            transformed = transform(pred, scale=scale)
            loss = full_loss(pred, self.boundaries, strikes, lows, highs, scale=scale)
            plot_polys(polys, self.boundaries, e, 'it', extra_points=[(strikes, lows), (strikes, highs)], title=str(loss.item()))

            polys = [np.polynomial.polynomial.Polynomial(transformed.detach().numpy()[i,:]) for i in range(transformed.shape[0])]
            plot_polys(polys, self.boundaries, e, 'tit',str(loss.item()))

            print('Loss:', loss)


            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


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
        for i in range(coeffs.shape[0]):
            return_grad[i] = x.item() ** i
        return return_grad * grad_output, None


class polyLoss(torch.autograd.Function):
      
    @staticmethod
    def forward(ctx, cond_coeffs, bounds):
        tol = 1.0e-5 # Tolerance for discarding imaginary roots

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
        coeff_ders, bounds_ders = ctx.saved_tensors
        # print(coeff_ders.mean())
        return coeff_ders, None

        
def plot_polys(polys, boundaries, num, name, title='', extra_points=None):
    # print("./"+name+str(num)+".png")
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
    # plt.vlines(boundaries, ymin = min(plot_points), ymax = max(plot_points), color = 'red')
    plt.title(title)
    plt.savefig("./"+name+str(num)+".png")
    plt.close()

def full_loss(coeffs, boundaries, strikes, lows, highs, scale=10000000):
    transformed = transform(coeffs, scale=scale)
    loss = (scale * polyLoss.apply(transformed, boundaries))
    print("Arb loss:", loss)
    if loss.item() <= 1.0e-7:
        print('yes!')
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
    # print(strikes.shape, lows.shape, highs.shape)
    # print('Lens: ', len(polys), len(boundaries))
    for strike, low, high in zip(strikes, lows, highs):
        while last_bound < len(boundaries) - 2 and boundaries[last_bound] < strike:
            last_bound += 1
        # print("--------")
        predicted = polys[last_bound](strike)
        # tester = polys_test[last_bound](strike)
        mid = (high + low) / 2.0
        out = (high - low) / 2.0
        err = (predicted - mid).abs()
        # print(predicted.item(), mid.item(), err.item(), (err + torch.max(err - out, zero)**2).item())
        # print(torch.autograd.grad(err, coeffs, retain_graph=True))
        # print(tester)
        # print(polys_test[last_bound])
        # print("-------------")
        loss = loss + 0.5 * err + torch.max(err - out, zero)**4 + torch.max(err - out, zero)**2 
    print('Mid loss', loss.item() / len(strikes))
    return loss / len(strikes)