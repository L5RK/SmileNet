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
    system[-4, 0:4] = torch.tensor([0,0,0,1])
    system[-3, -4:] = torch.tensor([0,0,0,1])
    system[-2, 0:4] = torch.tensor([0,0,1,0])
    system[-1, -4:] = torch.tensor([0,0,1,0])
    

    return system

def get_basis(system, knots, seeded_vecs, k_start, k_end):
    p1 = torch.tensor([0,1,2,3])
    p2 = torch.tensor([0,0,1,2])
    p3 = torch.tensor([0,0,0,1])
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
        wb = seeded_vecs[:,[i]] - Q @ (Q.T @ seeded_vecs[:,[i]])
        wb = wb -  basis @ (basis.T @ wb)
        wb = wb / torch.linalg.norm(wb)
        basis = torch.hstack([basis, wb])
    return basis, new_sys

def get_vector_seed(num_knots):
    dof = num_knots   # Subject to change -> Linear wings -4 for linear wings
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
def get_knots_perc(strikemin, strikemax, diffs):
    diff_oshape = diffs.shape
    diffs = diffs.reshape(-1)
    remaining = strikemax - strikemin
    knots = torch.ones(diffs.shape).double() * strikemin
    for i in range(diffs.shape[0]):
        knots[i:] = knots[i:] + remaining * diffs[i]
        remaining = remaining - remaining * diffs[i]
    return knots.reshape(diff_oshape)

def get_reverse_perc(strikemin, strikemax, knots):
    remaining = strikemax - strikemin
    prev = strikemin
    diffs = torch.zeros(knots.shape).double()
    for i in range(knots.shape[0]):
        # print(remaining)
        # print(knots[i]-prev)
        diffs[i] = (knots[i] - prev) / remaining
        remaining = (remaining - (knots[i] - prev))
        prev = knots[i]
    # test = get_knots_perc(strikemin, strikemax, diffs)
    # print("in test")
    # print(test - knots)
    # print(diffs)
    # print(strikemin, strikemax)
    # print(knots)
    # raise Exception("D")
    return diffs

def get_xstar(basis, knots, strikes, mids, ini_sys, low, spreads=None):
    # Pred - knot n is before strike n
    # Pred - strikes are sorted increasing
    # pred - knots are sorted increasing
    # print("going in ")
    # print(basis.shape, knots.shape, mids.shape, ini_sys.shape)
    # 0 , 1, 2, 3
    if spreads is None:
        spreads = torch.ones(mids.shape)
    mids = torch.sqrt(torch.max(torch.tensor(mids**2 - low),torch.tensor(0)))
    spreads = torch.diag(1/spreads)
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
    bsol  = torch.linalg.lstsq((eqs @ basis), mids.reshape(-1,1))[0]
    sol = basis @ bsol
    polys = [np.polynomial.polynomial.Polynomial(list(sol[:,0].detach().numpy()[4*i:4*i+4])) for i in range((sol.shape[0])//4)]
    # plot_polys([(lambda x, p=poly: (p(x) ** 2 + low) ** 0.5) for poly in polys], torch.hstack([knots[0]-(knots[1]-knots[0]), knots, knots[-1] + (knots[1]- knots[0])]), 0, 'before', extra_points=zip(strikes, mids + low))

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

    def __init__(self, num_knots, strikes, mids, low):
        super().__init__()
        # Blackbox layers
        # For now assume fixed strikes for each input
        dof = num_knots # Linear wings -> -4 degrees of freedom
        strike_knots = []
        count = 0
        # initial_knots = (torch.arange(num_knots)) * (strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1)+ (strikes[0] + strikes[1]) / 2
        strikes = torch.exp(strikes)
        initial_knots = (torch.arange(num_knots)) * ((strikes[-10] + strikes[-1])/2 - (strikes[10] + strikes[0]) / 2) / ((num_knots)- 1)+ (strikes[0] + strikes[10]) / 2
        # initial_knots = (torch.arange(num_knots)) * (( strikes[-1] + 0.1)- (strikes[0] -0.1)) / ((num_knots)- 1)+ (strikes[0] - 0.1)

        strikes = torch.log(strikes)
        initial_knots = torch.log(initial_knots)
        # initial_knots = strikes[[i * strikes.shape[0] // (num_knots+2) for i in range(1,num_knots+1)]]
        self.strike_low = strikes[0] - (strikes[-1] - strikes[0]) * 0.3
        self.strike_high = strikes[-1] + (strikes[-1] - strikes[0]) * 0.1

        # REMOVE THIS JUST FOR PLOTTING
        # self.strike_low = strikes[0]-0.2
        # self.strike_high = strikes[-1] + 0.5

        # print(self.strike_low, self.strike_high)
        # raise Exception("Deliberate")
        self.seeded_vecs = get_vector_seed(num_knots)
        self.base_system = get_system_matrix(num_knots)
        init_basis, ini_sys = get_basis(self.base_system, initial_knots, self.seeded_vecs, initial_knots[[0]], initial_knots[[1]])
        xstar = get_xstar(init_basis, initial_knots, strikes, mids, ini_sys, low)
        strike_num = strikes.shape[0]

        self.blackbox = torch.nn.Sequential(
            torch.nn.Linear(strike_num * 3 + 3, strike_num * 4 + 3), # 2*strikes for bid, ask + 1*strikes for strike pos + T for time til maturity + R for interest rate + S for current price
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 4 + 3, strike_num * 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 8, strike_num * 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(strike_num * 16, dof + (num_knots) + 1) #- len(strike_knots)),
        )
        self.relu = torch.nn.ReLU()
        self.knot_skeleton = initial_knots
        for i in self.blackbox:
            if not isinstance(i, torch.nn.LeakyReLU):
                i.weight = torch.nn.Parameter(torch.zeros(i.weight.shape).double() / 100)
                i.bias = torch.nn.Parameter(torch.zeros(i.bias.shape).double() / 100)
        new_bias = torch.zeros(self.blackbox[-1].bias.shape)
        new_bias[dof:-1] = get_reverse_perc(self.strike_low, self.strike_high, self.knot_skeleton)
        # new_bias[:dof] = xstar[:,0]
        # i_bias = torch.ones(num_knots).double() 
        # i_bias[0] += (strikes[0] - strikes[1]) / 2 - strikes[0]
        # i_bias = i_bias / ((strikes[-1] - (strikes[1] + strikes[0]) / 2) / (num_knots - 1))
        # new_bias[dof:] = torch.nn.Parameter(i_bias) 
        # print(new_bias[dof:])
        self.blackbox[-1].bias = torch.nn.Parameter(new_bias)
        self.scale = 1.0e8
        self.dof = dof
        self.num_knots = num_knots
        self.low = low
        self.memo = None

    def forward(self, x,):
        # TODO control as correction to lstsq
        raw = self.blackbox(x)
        control = raw[:,:self.dof]
        knots = raw[:,self.dof:-1]
        low_cont = self.relu(raw[0,-1])
        knots = torch.min(self.relu(knots), torch.tensor(0.99))
        # print(control.T)
        if self.memo is None:
            flatx = x.reshape(-1)
            his = flatx[:(flatx.shape[0]-3)//3]
            lows = flatx[(flatx.shape[0]-3)//3:2*(flatx.shape[0] - 3)//3] # TODO Add Weighted least squares
            self.low = torch.max(torch.min(lows) ** 2 / 2, torch.min((his + lows)/2)**2/4) 
            mids = (his + lows) / 2

            spreads = his-lows
            strikes = flatx[2*(flatx.shape[0] - 3)//3:-3]
            k_start = self.knot_skeleton.reshape(-1)[[0]]
            k_end = self.knot_skeleton.reshape(-1)[[-1]]
            init_basis, ini_sys = get_basis(self.base_system, self.knot_skeleton.reshape(-1), self.seeded_vecs, k_start, k_end)

            xstar = get_xstar(init_basis, self.knot_skeleton.reshape(-1), strikes,mids , ini_sys, self.low, spreads=spreads)
            self.memo=xstar
        else:
            xstar = self.memo
        smin = self.low + low_cont
        knots = get_knots_perc(self.strike_low.item(), self.strike_high.item(), knots) #+ self.knot_skeleton.reshape(knots.shape)
        # knots = self.knot_skeleton.reshape(1,-1)
        k_start = self.knot_skeleton.reshape(-1)[[0]]
        k_end = self.knot_skeleton.reshape(-1)[[-1]]
        translate, sys = get_basis(self.base_system, knots[0,:], self.seeded_vecs, k_start, k_end)
        control = control + xstar.reshape(1,-1)
        right_grad = torch.abs(translate[[-3],:] @ control.T)
        reduce = torch.max(right_grad/2.0, torch.tensor(1.0))
        control = control / reduce
        

        sol = translate @ control.T
        sol = getBlockForm(sol)
        bounds = torch.hstack([torch.tensor(self.strike_low).reshape(1,1), knots, torch.tensor(self.strike_high).reshape(1,1)])
        bounds = bounds.reshape(-1)

        return sol, bounds, smin, knots

    def train(self, data, strikes, epochs=1000, optimizer=None, inlr=None):
        if inlr is None: inlr =5.0e-7
        lows, highs = data[0, (data.shape[1] - 3)//3:2 * (data.shape[1] - 3) // 3], data[0,:(data.shape[1] - 3)//3]
        mids = (highs + lows) / 2
        mids_delta = mids[1:] - mids[:-1]
        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=inlr) # Was 5.0e-6 for Axel last succesful run
        stat = [1]
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda x: 0.995**(min(x, 1000))))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85)
        S = data[0,-3]
        T = data[0,-2]
        R = data[0,-1]
        losses = []
        delta_imp = 5.0e-2 * 1
        arb_imp = 1_000
        end_imp = 10
        bound_imp=1.0e-1
        raw_window = [0 for _ in range(10)]
        delta_window = [0 for _ in range(10)]
        arb_window = [0 for _ in range(10)]
        prev_raw = None
        prev_delta = None
        prev_arb = None
        wait_arb = 0
        wait_del = 0
        arb_tol = 1.0e-3
        for e in range(epochs):
            # Loss and prediction

            sol, bounds, low, knots = self.forward(data)
            # print('bounds vs ks')
            # print(bounds)
            # print(self.knot_skeleton)

            # print("Sol")
            # print(sol)
            ba = bounds
            ba[-1] = ba[-1]
            ba[0] = ba[0]

            polys = [np.polynomial.polynomial.Polynomial(sol.detach().numpy()[i,:])  for i in range(sol.shape[0])]
            gs = [g_func(sol[i,:], low) for i in range(sol.shape[0])]
            fs = [p_to_f(sol[p,:], low) for p in range(sol.shape[0])]
            if e%50 == 0:
                plottable_fs = [p_to_f(sol[p,:].detach(), low.detach().item()) for p in range(sol.shape[0])]
                plottable_gs = [g_func(sol[i,:].detach(), low.detach().item()) for i in range(sol.shape[0])]
                rnms = [f_g_to_rnm(f, g) for f, g in zip(plottable_fs, plottable_gs)]
                plot_polys(plottable_fs, ba.detach(), e, 'func','Plot of w(k)',extra_points=[(strikes, lows, 'Bids'), (strikes, highs, 'Asks')],labels=True, x_axis='Log-Moneyness (k)',y_axis='Total Variance')        
                plot_polys(plottable_gs, ba.detach(), e, 'tint','Plot of g(k)', x_axis='Log-Moneyness (k)', y_axis='g(k)')        
                plot_polys(rnms, ba.detach(), e, 'rnm','Plot of implied Risk Neutral Measure', x_axis = 'Log-Moneyness', y_axis='Probability Density')        
                plot_polys(polys,ba.detach(), e, 'raw','p',)

            # price_funcs = [lambda x, w=w: bsc_svi(x, w, S) for w in plottable_fs]
            # plot_polys(price_funcs, ba.detach().reshape(-1), e, 'prices')
            # plot_polys([p.deriv() for p in polys],ba.detach(), e, 'der','p',)
            # plot_polys([p.deriv(2) for p in polys],ba.detach(), e, 'der2','p',)


            # plot_polys([np.polynomial.polynomial.Polynomial(sol[p,:].detach()) for p in range(sol.shape[0])], bounds.reshape(-1), e, 'poly')

            g_loss = [lambda x, g=i: gloss_func(g, x, arb_tol) for i in gs]
            real_gloss = [lambda x, g=i: real_gloss_func(g, x) for i in gs]
            arb_loss = torch.tensor(0.0)
            real_arb = torch.tensor(0.0)
            for g, r, a, b in zip(g_loss, real_gloss, bounds[:-1], bounds[1:]):
                arb_loss = arb_loss + diff_integ(g, a, b)
                real_arb = real_arb + diff_integ(r, a, b)

            # area = torch.tensor(0.0)
            # for r, a, b in zip(rnms, ba[:-1].detach(), ba[1:].detach()):
            #     area += diff_integ(r, a, b, delta=1.0e-5).item()
            
            # print('Area',area)
            raw_loss, delta_loss, delta_first, delta_last, mape = get_observe_losses(fs, bounds[1:-1], strikes, lows, highs, delta_imp)
            wait_arb += 1
            wait_del += 1
            if prev_raw is None:
                prev_raw = raw_loss
                prev_delta = delta_loss
                prev_arb = arb_loss

            raw_chng = 1 - (raw_loss / prev_raw).item() if raw_loss != 0 else 0
            delta_chng = 1 - (delta_loss / prev_delta).item()
            arb_change = 1 - (arb_loss / prev_arb).item()

            prev_raw = raw_loss
            prev_delta = delta_loss
            prev_arb = arb_loss
            raw_window = raw_window[1:] + [raw_chng]
            delta_window = delta_window[1:] + [delta_chng]
            arb_window = arb_window[1:] + [arb_change]
     
            print(delta_chng, raw_chng, arb_change)
            if sum(raw_window) > sum(arb_window) and arb_loss != 0 and wait_arb > 10:
                # arb_tol = arb_tol * 1.0005
                arb_imp = arb_imp * 1.05
                print('arb imp changed to', arb_imp)
                wait_arb = 0

            if sum(raw_window) < sum(delta_window) and raw_loss != 0 and wait_del > 10:
                delta_imp = delta_imp / 1.025
                delta_window = [0 for _ in range(10)]
                print('Delta imp changed to', delta_imp)
                wait_del = 0
            # if sum([1 if x>=0 else 0 for x in  raw_window])==10 and sum(raw_window) < 0.05:
            #     stat[0] = stat[0] * 1.01
            #     print('stat raise', stat[0])

            if arb_loss == 0:
                arb_imp = max(100, arb_imp / 1.0001)
                # arb_tol = min(1.0e-5, arb_tol/1.001)
            bounds_loss = raw_loss
            bounds_loss = bounds_loss * bound_imp
            arb_loss = arb_imp * arb_loss 
            delta_loss = delta_loss * delta_imp

            if e%1==0:
                print('Epoch:',e)
                print('Arb loss', arb_loss.item())
                print('Real arb loss', real_arb.item())
                print('Bounds loss:', bounds_loss.item())
                print('MAPE: ', mape.item())
                print('Delta: ', delta_loss.item())
            if mape == 0 and real_arb.item() == 0:
                print('Breaking and plotting at epoch',e,'with bounds loss', bounds_loss, 'and arb loss', arb_loss)
                plottable_fs = [p_to_f(sol[p,:].detach(), low.detach().item()) for p in range(sol.shape[0])]
                plottable_gs = [g_func(sol[i,:].detach(), low.detach().item()) for i in range(sol.shape[0])]
                rnms = [f_g_to_rnm(f, g) for f, g in zip(plottable_fs, plottable_gs)]
                plot_polys(plottable_fs, ba.detach(), e, 'func','Plot of w(k)',extra_points=[(strikes, lows, 'Bids'), (strikes, highs, 'Asks')],labels=True, x_axis='Log-Moneyness (k)',y_axis='Total Variance')        
                plot_polys(plottable_gs, ba.detach(), e, 'tint','Plot of g(k)', x_axis='Log-Moneyness (k)', y_axis='g(k)')        
                plot_polys(rnms, ba.detach(), e, 'rnm','Plot of implied Risk Neutral Measure', x_axis = 'Log-Moneyness', y_axis='Probability Density')        
                plot_polys(polys,ba.detach(), e, 'raw','p',)
                break

            print('GRAD\n',torch.autograd.grad(arb_loss, sol, retain_graph=True)[0][-3,:])
            loss = arb_loss + raw_loss + delta_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)
        plt.plot(losses)
        plt.savefig('./loss.png')
        plottable_fs = [p_to_f(sol[p,:].detach(), low.detach().item()) for p in range(sol.shape[0])]
        plottable_gs = [g_func(sol[i,:].detach(), low.detach().item()) for i in range(sol.shape[0])]
        ba[0] = -np.inf
        ba[-1] = np.inf
        w = lambda k, buckets=ba, fs=plottable_fs:  print(np.where((buckets[:-1] < k) & (buckets[1:] > k))) if False else fs[np.where((buckets[:-1] < k) & (buckets[1:] > k))[0][0]](k) 
        return w

    def pred_as_function(self, x):
        sol, boundaries, low, knots = self.forward(x)
        fs = [p_to_f(sol[p,:].detach().numpy(), low.detach().numpy()) for p in range(sol.shape[0])]
        buckets = np.array([-np.inf] + [i for i in knots.detach().numpy()] +[np.inf])
        w = lambda k, buckets=buckets, fs=fs: print(np.where((buckets[:-1] < k) & (buckets[1:] > k))) if False else fs[np.where((buckets[:-1] < k) & (buckets[1:] > k))[0][0]](k) 

        return w

    def single_run(self, data, strikes):
        lows, highs = data[0, (data.shape[1] - 3)//3:2 * (data.shape[1] - 3) // 3], data[0,:(data.shape[1] - 3)//3]
        mids = (highs + lows) / 2
        mids_delta = mids[1:] - mids[:-1]
        S = data[0,-3]
        T = data[0,-2]
        R = data[0,-1]
        losses = []
        sol, bounds, low = self.forward(data, scale)
        ba = bounds
        ba[-1] = ba[-1]
        ba[0] = ba[0]

        polys = [np.polynomial.polynomial.Polynomial(sol.detach().numpy()[i,:])  for i in range(sol.shape[0])]
        gs = [g_func(sol[i,:], low.item()) for i in range(sol.shape[0])]
        fs = [p_to_f(sol[p,:], low.item()) for p in range(sol.shape[0])]
        plottable_fs = [p_to_f(sol[p,:].detach(), low.detach().item()) for p in range(sol.shape[0])]
        plottable_gs = [g_func(sol[i,:].detach(), low.detach().item()) for i in range(sol.shape[0])]

        rnms = [f_g_to_rnm(f, g) for f, g in zip(plottable_fs, plottable_gs)]

        plot_polys(plottable_fs, ba.detach(), e, 'func','p',extra_points=[(strikes, lows), (strikes, highs), (strikes, [low.detach().item() for i in strikes])])        
        plot_polys(plottable_gs, ba.detach(), e, 'tint','t')        
        plot_polys(rnms, ba.detach(), e, 'rnm','r')        

        plot_polys([np.polynomial.polynomial.Polynomial(sol[p,:].detach()) for p in range(sol.shape[0])], bounds.reshape(-1), e, 'poly')

        g_loss = [lambda x, g=i: (torch.min(g(x), torch.tensor(0.0)))**2 for i in gs]
        arb_loss = torch.tensor(0.0)
        for g, a, b in zip(g_loss, bounds[:-1], bounds[1:]):
            arb_loss += diff_integ(g, a, b)
        arb_loss = 10_000 * arb_loss 
        print('Arb loss', arb_loss.item())
        bounds_loss, d, mape = get_observe_losses(fs, bounds[1:-1], strikes, lows, highs)
        bounds_loss = 1.0 * bounds_loss

        print('Bounds loss:', bounds_loss.item())
        print('MAPE: ', mape)
        print('Delta: ', d)
        plt.plot(losses)
        plt.savefig('./loss.png')
        return sol, bounds, low

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



def d_sub(lk, w):
    wsk= lk + 1 
    return -lk / np.sqrt(w(wsk)) - np.sqrt(w(wsk))/2

def rn_tv(k, w, g):
    # norm is not proven or anything

    k = k - 1.0
    # print(np.exp(-d_sub(k,w)**2 / 2))
    # print(w(k+1))
    return np.exp(-d_sub(k, w)**2 / 2) * g(k + 1) / (np.sqrt(2 * np.pi * w(k+1)))

def transform(sol, k, scale=1):# Remember to add k as required parameter
    # Assume sol is of form a0, b0, c0, d0, a1, b1, ...
    # w(x) of form sqrt(p(x)^2 + k)
    # aiming to remove (g(x) f(x)^4 )^2 = 0
    # This is of form ((f(x)^2 - 1/2 ax)^ 2) - 1/16 a^2 f^2) ^2 >= f(x)^2(-3/4 a^2 + f(x)^2(b+c))^2
    # where:
    #   f(x) = p(x)^2 + k
    #   a = p'(x)p(x)
    #   b = p(x)p''(x)
    #   c = p'(x)^2
    
    transforms = []
    sol = sol
    sol_degree = sol.shape[1]
    max_degree = sol_degree * 6 + (sol_degree - 1) * 2 # Corresponding to f^4 a^4 = O(p(x)^2) O( p'(x)^4 p(x)^4)
                                                       #                          = O(p(x)^6 p'(x)^2
    preservingPolyMul = lambda x, y: differentiablePolyMul.apply(x, y, max_degree)
    preservingPolyAdd = lambda x, y: differentiablePolyAdd.apply(x, y, max_degree)
    preservingPolySub = lambda x, y: differentiablePolySub.apply(x, y, max_degree)
    for i in range(0, sol.shape[0]):


        # preservingTensorPoly = lambda x: convertTensorPoly.apply(x, max_degree)

        p = sol[i,:]
        ps = differentiablePolyDeriv.apply(p)
        pss = differentiablePolyDeriv.apply(ps)

        # Now define a, b, c, f as in intro
        a = preservingPolyMul(p, ps)
        b = preservingPolyMul(p, pss)
        c = preservingPolyMul(ps, ps)
        f2 = preservingPolyAdd(p, k) # equal to f(x)^2 = (p(x)^2 + k)

        x = torch.tensor([0,1]) # Remember to account for translation
        condL = preservingPolySub(f2, 0.5 * a)
        condL = preservingPolyMul(condL, condL)
        a2f2 = preservingPolyMul(preservingPolyMul(-1.0/16 * a, a), f2)
        condL = preservingPolySub(condL, a2f2)
        condL = preservingPolyMul(condL, condL)
        # Above corresponds to ((f(x)^2 - 1/2a)^2 - 1/16 a^2f(x)^2)^2

        condR = preservingPolyAdd(b, c)
        condR = preservingPolyMul(f2, condR)
        condR = preservingPolySub(condR, 3.0/4 * preservingPolyMul(a,a))
        condR = preservingPolyMul(condR, condR)
        cond = preservingPolySub(condL, condR)

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
    def forward(ctx, coeff1, coeff2, preserve_degree=None):
        # Scale is tuple (a, b) st result = poly1 * a + poly2 * b
        poly1 = np.polynomial.polynomial.Polynomial(coeff1.detach().numpy())
        poly2 = np.polynomial.polynomial.Polynomial(coeff2.detach().numpy())
        result = torch.tensor(np.polymul(poly1, poly2)[0].coef)
        if preserve_degree is None:
            preserve_degree = result.shape[0]
        ctx.save_for_backward(coeff1, coeff2, torch.tensor(preserve_degree))
        if preserve_degree is not None:
            ret = torch.zeros((preserve_degree + 1,))
            ret[:result.shape[0]] = result
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
        xder = torch.tensor(poly.deriv()(x))
        for i in range(coeffs.shape[0]):
            return_grad[i] = torch.sum(x ** i * grad_output)
        return return_grad, xder * grad_output

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

def plot_polys(polys, boundaries, num, name, title='', extra_points=None, labels=False, x_axis='', y_axis=''):
    boundaries = boundaries.detach()
    boundary_spaces = [torch.tensor(np.linspace(x,y,num=100)) for x, y in zip(boundaries[:-1],boundaries[1:])]
    plot_points = np.hstack([poly(space) for poly, space in zip(polys, boundary_spaces)])
    # for i in range(len(boundary_spaces)):
    points = np.hstack([i.detach() for i in boundary_spaces])
    plt.figure(figsize=(20,20))
    plt.plot(points.reshape(-1), plot_points.reshape(-1))

    if extra_points is not None:
        if not labels:
            for x, y in extra_points:
                plt.scatter(x, y)
        else:
            for x, y, z in extra_points:
                plt.scatter(x,y,label=z, s=20)
            plt.legend(loc='upper right')
    plt.vlines(boundaries, ymin = min(plot_points.reshape(-1)), ymax = max(plot_points.reshape(-1)), color = 'red')
    plt.hlines([0.0], xmin=boundaries[0], xmax=boundaries[-1], color='black')
    plt.suptitle(title, fontsize=24)
    if labels:
        plt.legend(loc='upper right', prop={'size':24})
    plt.xlabel(x_axis, fontsize=24)
    plt.ylabel(y_axis, fontsize=24)
    plt.savefig("./"+name+str(num)+".png")
    plt.close()
    # for i in polys:
    #     print(i)
    # for j in boundaries:
    #     print(j)
    # raise Exception('Deliberate')

def get_observe_losses(fs, boundaries, strikes, lows, highs, delta_imp):
    # Pred: boundaries, strikes sorted
    # print("Testing:: ")
    # polys_test = [np.polynomial.polynomial.Polynomial(list(reversed(coeffs[:,0].detach().numpy()[4*i:4*i+4]))) for i in range((coeffs.shape[0])//4)]
    last_bound = 0
    loss = torch.tensor(0.0).double()
    zero = torch.tensor(0.0).double()

    # Delta first and delta last for early stopping
    delta_first = None
    delta_last = None


    last_mid = None      
    last_pred = None
    errs = []
    delta_loss = torch.tensor(0.0).double()
    # MAPE
    mape = 0
    epsilon = 1.0e-1
    # DELTA
    bounds_norm = torch.exp(strikes[-1]) - torch.exp(strikes[0])
    outs_norm = torch.tensor(0.0)
    for strike, low, high in zip(strikes, lows, highs):
        while last_bound < len(boundaries) and boundaries[last_bound] < strike:
            last_bound += 1
        predicted = fs[last_bound](strike)
        mid = (high + low) / 2.0
        if last_mid is not None:
            mid_delta = (mid - last_mid) 
            func_delta = (predicted - last_pred) 
            delta_loss = delta_loss + ((mid_delta - func_delta) / mid_delta)**2 * (torch.exp(strike) - torch.exp(last_strike))
            if delta_first is None:
                delta_first = max(lows[1] - lows[0] - func_delta, 0)
        last_mid = mid
        last_pred = predicted
        last_strike = strike
        out = (high - low) / 2.0  # how far is bid/ask
        err = (predicted - mid).abs() 
        mape += (torch.max(err-out, zero) / mid)
        outs_norm += (out ** 0.5)
        loss = loss + ((torch.max(err - out * (1- epsilon), zero) / mid))** 2 / (out ** 0.5)
    delta_last =  max(func_delta - (highs[1] - highs[0]), 0)
    return loss / outs_norm, delta_loss / bounds_norm, delta_first, delta_last, mape / len(strikes)

def f(x, p, k):
    return torch.sqrt(differentiablePolyEval.apply(p, x) ** 2+ k)

def gloss_func(g, x, arb_tol):
    gs = g(x)
    x = torch.abs(torch.min(gs - arb_tol, torch.tensor(0.0)))
    y = torch.abs(torch.min(gs, torch.tensor(0.0)))
    return 1.0 * x**2 + y + y**2

def real_gloss_func(g, x):
    gs = g(x)
    y = torch.abs(torch.min(gs, torch.tensor(0.0)))
    return y


def g_func(p, lo):

    # get a(x) = p(x)p'(x) as a function    
    adeg = (p.shape[0] - 1)+ (p.shape[0] - 2)
    ap = differentiablePolyMul.apply(p, differentiablePolyDeriv.apply(p), adeg)
    psub_back = differentiablePolyDeriv.apply(p)
    psub2_back = differentiablePolyDeriv.apply(psub_back)
    psub = lambda x: differentiablePolyEval.apply(psub_back, x)
    psub2 = lambda x: differentiablePolyEval.apply(psub2_back, x)
    p_eval = lambda x: differentiablePolyEval.apply(p, x)
    a = lambda x: differentiablePolyEval.apply(ap, x)
    p2 = lambda x: differentiablePolyEval.apply(p, x) ** 2
    w = lambda x: f(x, p, lo)
    term1 = lambda x: (1 - a(x)*x/(2 * w(x) ** 2)) ** 2
    term2 = lambda x: (1 / w(x) + 1/4) * (a(x) ** 2 / (4 * w(x) ** 2) )
    term3 = lambda x: 0.5 * ((p_eval(x) * psub2(x) + psub(x)**2) / w(x) - a(x) ** 2 / (w(x) ** 3))
    return lambda x: term1(x) - term2(x) + term3(x)

def p_to_f_sub(p, low):
    ap = differentiablePolyMul.apply(p, differentiablePolyDeriv.apply(p))
    return lambda x: differentiablePolyEval.apply(p, x) * differentiablePolyEval.apply(ap, x) / f(x, p, low)

def p_to_f(p, low):
    return lambda x: torch.sqrt(differentiablePolyEval.apply(p, x) ** 2 + low)

def p_to_f_plot(p, low):
    return lambda x: torch.sqrt(np.polynomial.polynomial.Polynomial(p.detach().numpy())(x)**2 + low)

def f_g_to_rnm(f, g):
    dsub = lambda x: -x / (f(x) ** 0.5) - f(x)**0.5 / 2
    return lambda x: (g(x) / (np.sqrt(np.pi * 2 * f(x))) * np.exp(-dsub(x) ** 2 / 2))

def d_plus(k, w):
    return -k / torch.sqrt(w(k)) + torch.sqrt(w(k)) / 2
def d_sub(k,w):
    return -k / torch.sqrt(w(k)) - torch.sqrt(w(k)) / 2
def bsc_svi(k, w, S):
    norm = lambda x: differentiableNormCdf.apply(x)
    return S * (norm(d_plus(k, w)) - torch.exp(k) * norm(d_sub(k, w)))
def plot_vertical_spreads(ps, low, bounds, filename, num):
    linspace = []
    lower = []
    value = []
    upper = []

    for p, a, b in zip(ps, bounds[:-1], bounds[1:]):
        if a < 0:
            continue
        if b > 1.2:
            b = 15
        wsub = p_to_f_sub(p, low)
        w = p_to_f(p, low)
        cond = lambda x: wsub(x) / torch.sqrt(w(x))
        bound = torch.tensor(np.linspace(a, b, 50))
        value += [cond(bound)]
        linspace += [bound]
        upper += [-norm.cdf(d_sub(-bound, w)) / norm.pdf(d_sub(bound, w))]
        lower += [norm.cdf(d_plus(bound, w)) / norm.pdf(d_plus(bound, w))]
    linspace = np.hstack([linspace]).reshape(-1)
    upper = np.hstack([upper]).reshape(-1)
    lower = np.hstack([lower]).reshape(-1)
    value = np.hstack([value]).reshape(-1)

    # plt.plot(linspace, upper, label='Upper')
    plt.plot(linspace, value, label='Value')
    plt.plot(linspace, lower, label='Lower')
    plt.legend(loc = 'upper left')
    plt.savefig(filename + str(num)+".png")

def diff_integ(f,a,b, delta=1.0e-3):
    # Computes integral of f between a and b with step_size delta
    rng = torch.arange(int((b - a) / delta)) * delta + a
    return torch.sum((f(rng[1:]) + f(rng[:-1]))/2 * delta)


def removeArb(control, translate, boundaries, sl, sh, low, sys=None):
    lr = 1
    gamma = 1.1
    tol=1.0e-6
    max_epochs = 100
    scale=1
    # Want to apply normalisation s.t x_min k + a = 0.9
    #                                 x_max k + a = 1.1
    # Then we get k = 0.2 / (x_min - x_max), a = 0.9
    # Normalize data:

    # x = translate @ torch.eye(4).double()

    ## USEFUL DON'T DELETE
    # for i in range(x.shape[0]):
    #     l = list(x[i,:])
    #     s = "a_{"+str(i//4) +str(i%4)+"} = "
    #     for j in range(len(l)):
    #         s += "+ "+str(round(l[j].item(), 7))+"c_"+str(j)+" "
    #     print(s)
    # for i in range(boundaries.shape[1]+1):
    #     print("p_{%s} = a_{%s0} + a_{%s1} x + a_{%s2} x^2 + a_{%s3}x^3" % (str(i), str(i), str(i), str(i), str(i)))
    # print(boundaries)
    # print(low)
    # raise Exception("Deliberate")
    for e in range(max_epochs):
        # print(control)
        sol = translate @ control
        sol = getBlockForm(sol)
        polys = [np.polynomial.polynomial.Polynomial(sol.detach().numpy()[i,:])  for i in range(sol.shape[0])]
        plot_polys(polys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'raw_poly','p')        
        plot_polys([poly.deriv(1) for poly in polys], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'raw_poly_sub','p')        
        plot_polys([poly.deriv(2) for poly in polys], torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'raw_poly_sub2','p')        

        gs = [g_func(sol[i,:], low) for i in range(sol.shape[0])]
        plottable_gs = [(lambda x, y=g: y(x).detach()) for g in gs]
        plot_polys(plottable_gs, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]).detach(), e, 'tint','t')        
    
        fs = [p_to_f(sol[p,:].detach(), low) for p in range(sol.shape[0])]
        plot_polys(fs, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]).detach(), e, 'func','p')        
        gmins = [lambda x, g=i: -1* torch.min(g(x) - 1.0e-6, torch.tensor(0.0)) for i in gs]
        arb_loss = torch.tensor(0.0)
        bounds = torch.hstack([torch.tensor(sl).reshape(1,1), boundaries, torch.tensor(sh).reshape(1,1)])
        for g, a, b in zip(gmins, bounds[0,:-1], bounds[0,1:]):

            arb_loss = arb_loss + diff_integ(g, a, b)

        # loss = polyLoss.apply(transformed, boundaries, sl, sh) # + scale * polyLoss.apply(sol, boundaries, sl, sh)
        ##########################
        print('Arb loss', arb_loss, e)
        # loss = loss + scale * polyLossT(polys, boundaries, sl, sh) # + scale * polyLoss.apply(sol, boundaries, sl, sh)
        print('Total loss',loss)
        ##########################
        # ppolys = [np.polynomial.polynomial.Polynomial(polys.detach().numpy()[i,:])  for i in range(polys.shape[0])]
        # plot_polys(ppolys, torch.hstack([torch.tensor(sl) , boundaries.reshape(-1), torch.tensor(sh)]), e, 'pint','v')        

        if loss < tol:
            break

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