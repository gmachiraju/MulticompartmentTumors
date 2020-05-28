import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.ticker as mtick
import pickle
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from itertools import cycle
import matplotlib.patches as mpatches
np.random.seed(3) 

#---------------
# Main methods
#---------------

def plot_sens_shed(X, Ye, Yne, t, titleStr, xlab, ylab, qb_ex, Ptot, pname, param, strformat):

    plt.style.use('grayscale')
    font = {'family': 'normal','weight': 'normal','size': 17}
    plt.rc('font', **font)
    
    colors = ["#7a7a7a", "#7a7a7a", "#7a7a7a", "#7a7a7a", "#7a7a7a"]
    colors2 = ["#000000", "#000000", "#000000", "#000000", "#000000"]
    markers = [10, "", "", "", "o"]
    starts = [0, 0, 0, 0, 300]
    ls = ["-", "--", "-.", ":", "-"]
    
    pname_title = pname
    if "{" in pname:
        pname_title = "t"

    if pname_title == "t" or pname_title == "$u_H$": 
        fig = plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
    else:
         fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

    for i in range(X.shape[1]):
        plt.loglog(X[:, i], Ye[:, i], linewidth=2, linestyle=ls[i], color=colors[i], markevery=(starts[i], 600), marker=markers[i], markersize=8)
        
    for i in range(X.shape[1]):
        plt.loglog(X[:, i], Yne[:, i], linewidth=2, linestyle=ls[i], color=colors2[i], markevery=(starts[i], 600), marker=markers[i], markersize=8)
        
    ax = fig.gca()
    ax.grid(False)
    ax.grid(False)
    
    if pname == "$\Phi$ or $\Psi$":
        labels = [r"$\Phi$" + " = " + strformat % i for i in param] + [r"$\Psi$" + " = " + strformat % i for i in param]
    else:
        labels = [pname + " = " + strformat % i for i in param] 
        labels = labels + labels
    
    legend_title = "Proliferative" + " "*14 + "Necrotic"
    legend_title2 = "Proliferative (gray)\nNecrotic (black)"
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim((1, 1e7-1))
    plt.axis(option="tight")
    plt.title(titleStr)
    
    if pname_title == "t" or pname_title == "$u_H$":
        plt.legend(labels, title=legend_title2, ncol=1, loc=(1.04,0))
        fig.tight_layout()
    else:        
        plt.legend(labels, title=legend_title, ncol=2)

    plt.savefig("spatial_figs/shed_sens_%s.png" % pname_title)
    plt.show()


def plot_sens_grow(X, Ye, Yne, t, titleStr, xlab, ylab, param, paramName, strformat):

    plt.style.use('grayscale')
    font = {'family': 'normal', 'weight': 'normal', 'size': 17}
    plt.rc('font', **font)
    colors = ["#7a7a7a", "#7a7a7a", "#7a7a7a", "#7a7a7a", "#7a7a7a"]
    colors2 = ["#000000", "#000000", "#000000", "#000000", "#000000"]
    markers = [10, "", "", "", "o"]
    starts = [0, 0, 0, 0, 300]
    ls = ["-", "--", "-.", ":", "-"]

    labels = [paramName + " = " + strformat % i for i in param] 
    labels = labels + labels
    legend_title = "Proliferative"
    legend_title2 = "Proliferative (gray)\nNecrotic (black)"
    
    fig = plt.figure(num=None, figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

    for i in range(X.shape[1]):
        plt.semilogy(X[:, i], Ye[:, i], linewidth=3, linestyle=ls[i], markevery=(starts[i], 600),
                     markersize=8, marker=markers[i], color=colors[i])
    
    for i in range(X.shape[1]):    
        plt.semilogy(X[:, i], Yne[:, i], linewidth=3, linestyle=ls[i], markevery=(starts[i], 600),
                     markersize=8, marker=markers[i], color=colors2[i])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax = fig.gca()
    ax.grid(False)    
    
    plt.axis([0, t[-1], 1,  5 * 10**8])
    plt.title(titleStr)
    plt.legend(labels, title=legend_title2, ncol=1, loc=(1.04,0))
    fig.tight_layout()
    
    plt.savefig("spatial_figs/grow_sens_%s.png" % paramName)
    plt.show()


def plot_compete(X, Y, t, Ntot, basal, titleStr, xlab, ylab, log_flag=None):

    plt.style.use('grayscale')
    font = {'family': 'normal', 'weight': 'normal', 'size': 17}
    plt.rc('font', **font)
    
    colors = ["#000000","#000000","#7a7a7a","#7a7a7a"]
    markers = [10, "o", "", "", ""]
    starts = [0, 300, 0, 0]
    ls = ["-", "-", "--", ":"]

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

    if log_flag == None or log_flag == "semilogy":
        for i in range(X.shape[1]): 
            plt.semilogy(X[:, i], Y[:, i], linewidth=3, markersize=10, linestyle=ls[i], marker=markers[i], markevery=(starts[i], 600), color=colors[i])
            plt.xlim((0,12))

    elif log_flag == "loglog":
        for i in range(X.shape[1]):
            plt.loglog(X[:, i], Y[:, i], linewidth=3, markersize=10, linestyle=ls[i], marker=markers[i], markevery=(starts[i], 600), color=colors[i])
            plt.xlim((1, 1e7-1))
    
    plt.axis(option="tight")
    ax = fig.gca()
    ax.grid(False)
    labels = ["EC", "Non-EC", "Non-EC ($\Psi$)", "non-EC ($t_{1/2}$)"]     

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titleStr)
    plt.legend(labels)
    plt.savefig("spatial_figs/compete_%s.png" % titleStr)
    plt.show()


def plot_compete_grid(X, Y, t, basal, titleStr, xlab, ylab, experiment=None, log_flag=None):

    plt.style.use('grayscale')
    font = {'family': 'normal', 'weight': 'normal', 'size': 17}
    plt.rc('font', **font)

    markers = ["d", "s"] 
    starts = [0, 300] + [150]*(X.shape[1]-2)
    
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

    for i in range(X.shape[1]):
        if i < 2:
            col = "#000000"
            mark = markers[i]
            me = (starts[i], 600)
            ms = 8
            if i ==0:
                lab = "EC"
            elif i ==1:
                lab = "Non-EC"
        else:
            col = "gray"
            mark = None 
            lab = '_nolegend_'
            
        if log_flag == "semilogy":
            axes = plt.semilogy(X[:, i], Y[:, i], linewidth=1, markersize=ms, linestyle='-', marker=mark, markevery=me, color=col, alpha=0.6, label=lab)
        elif log_flag == "loglog":
            axes = plt.loglog(X[:, i], Y[:, i], linewidth=1, markersize=ms, linestyle='-', marker=mark, markevery=me, color=col, alpha=0.6, label=lab) 
            
    # fill between lines
    plt.fill_between(X[:,0], Y[:, 2], Y[:, 3], color='#26235B', alpha='0.1')
    plt.fill_between(X[:,0], Y[:, 4], Y[:, 5], color='#A22160', alpha='0.1')
    plt.fill_between(X[:,0], Y[:, 6], Y[:, 7], color='#E4682A', alpha='0.1')
    
    # hori results
    maxIter = len(t)
    Phori, qhori = calc_hori(maxIter)
    
    if log_flag == "semilogy":
        plt.semilogy(X[:, 1], qhori, color="#000000", linewidth=3, label="Hori et al") # used to be red
    if log_flag == "loglog":
        plt.loglog(X[:, 1], qhori, color="#000000", linewidth=3, label= "Hori et al") # used to be red
    
    plt.axis(option="tight")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titleStr)
    
    if log_flag == "semilogy":
        plt.xlim((0,12))
    elif log_flag == "loglog":
        plt.xlim((1, 1e7-1))
                
    plt.legend()

    ax = fig.gca()
    ax.grid(False)
    plt.savefig("spatial_figs/compete_grid_%s.png" % titleStr)
    plt.show()
    
    
def plot_predicted(X, Y, titleStr, xlab, ylab, log_flag=None, rescale=False):

    plt.style.use('grayscale')
    font = {'family': 'normal', 'weight': 'normal', 'size': 17}
    plt.rc('font', **font)
    colors = ["#d9d9d9", "#bdbdbd", "#969696", "#636363", "#252525", "#000000"]
    markers = ["s", "o", "^", "d", "X", "<"]
    
    plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

    if log_flag == None or log_flag == "semilogy":
        for i in range(Y.shape[1]): 
            plt.semilogy(X[:, i], Y[:, i], linewidth=1, color="k")
    elif log_flag == "loglog":
        for i in range(Y.shape[1]):
            plt.loglog(X[:, i], Y[:, i], linewidth=1, color="k")
    
    plt.axis(option="tight")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titleStr)
    plt.legend()
    plt.show()


def plot_model_select(kvs, O2maxs, aEnd, nVec):
    
    plt.style.use('grayscale')
    font = {'family': 'normal', 'weight': 'normal', 'size': 20}
    plt.rc('font', **font)
    
    # setup
    sns.set_style("whitegrid")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # plane for cutoff
    xx = [kvs[0], kvs[-1]]
    yy = [O2maxs[0], O2maxs[-1]]
    xx, yy = np.meshgrid(xx, yy)
    
    xx = np.reshape(xx, (2,2))
    yy = np.reshape(yy, (2,2))    
    zz = np.reshape(np.log10(np.asarray([1e7, 1e7, 1e7, 1e7])), (2,2))
    
    ax.plot_surface(xx, yy, zz, alpha=0.2, zorder=0.8)
    
    # Make the surface plot
    X, Y = np.meshgrid(kvs, O2maxs)
    Z = np.array(aEnd)
    n,m = X.shape
    
    X = np.reshape(X, (n,m))
    Y = np.reshape(Y, (n,m))
    Z = np.reshape(Z, (n,m))
    Zn = np.reshape(nVec, (n,m))

    surf = ax.plot_surface(X, Y, np.log10(Z), cmap=grayscale_cmap("viridis"), linewidth=0.2, alpha=1, zorder=0.3)
    ax.zaxis.set_rotate_label(False) 
    
    # get stats on mesh runs
#     print(Zn)
#     print(Z)
#     print(X[0])
#     print(Y)
    
    # Add color bar which maps n-values to colors
    m = cm.ScalarMappable(cmap=grayscale_cmap("viridis")) # cm.bone
    m.set_array(Zn)
    cbar = plt.colorbar(m, shrink=0.3, aspect=5, pad=-0.08) 
    cbar.ax.set_ylabel('Number of compartments with growth', rotation=90)
    
    # format axes
    ax.set_xlabel(r'Vascularization rate, $k_V$ (mm/day)', labelpad=18)
    ax.set_ylabel(r'$C_0$ (%)', labelpad=14)
    ax.set_zlabel(r'Population at $t_{end}$ ($log_{10}$ cells)', rotation=90, labelpad=12)
    ax.set_title('Tumor growth selection using parameter mesh')

    plt.xticks(np.arange(kvs[0], kvs[-1]+0.1, 0.1))
    plt.yticks(np.arange(O2maxs[0], O2maxs[-1]+1, 2))
    ax.view_init(20, 240)
    fig.set_figwidth(12)
    fig.set_figheight(8)
    plt.savefig('spatial_figs/surface.png')
    plt.show()


def plot_comp_bar(PN, PNtot, t, n, titleStr, xlab, ylab, PN_flag=False, shed_flag=False, dtype=None):

    pal = sns.color_palette("cubehelix", 10) 
    font = {'family': 'normal','weight': 'normal','size': 17}
    plt.rc('font', **font)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100,facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    labels = [r"$T_{%s}$" % i for i in range(1,11)]
    plt.stackplot(t, PN.T, colors=pal, alpha=0.4, labels=labels)
    plt.yscale("log")
    plt.autoscale(enable=True, axis='x', tight=True)

    if shed_flag == True:
        plt.xscale("log")
        loc = 2
        if PN_flag == "P":
            plt.xlim(left=1e1) #1e4
            plt.ylim(bottom=1e-2, top=1e4)
        elif PN_flag == "N":
            plt.xlim(left=1e1)
            plt.ylim(bottom=1e-2, top=1e4)
    else:
        loc = 2
        plt.xlim((0, 12))
        plt.ylim(bottom=1e0, top=3e7)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[::-1], labels[::-1], loc=2,
                       title='Compartment', fontsize="medium", prop={'size': 12})

    plt.setp(legend.get_title(), fontsize='medium')
    
    ax = fig.gca()
    ax.grid(False)

    plt.title(titleStr)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig('spatial_figs/stackplot_%s.png' % titleStr)
    plt.show()


def plot_comp_rate(kBs, kDs, n):

    plt.style.use('grayscale')

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig = plt.figure(num=None, figsize=(9, 6), dpi=100, facecolor='w', edgecolor='k')
    ind = np.arange(n) + 1
    indstr = [r"$T_{" + str(i) + "}$" for i in ind]
    w = 0.5
    
    # reverse lists
    kBs = list(reversed(kBs))
    kDs = list(reversed(kDs))
    ind = np.array(list(reversed(list(ind))))
    indstr = list(reversed(list(indstr)))
    kGs = list(np.array(kBs) - np.array(kDs))

    p1 = plt.plot(ind, kBs, '-ko', linewidth=3, markersize=8)
    p2 = plt.plot(ind, kDs, '-k^', linewidth=3, markersize=8)
    p3 = plt.plot(ind, kGs, '-ks', linewidth=3, markersize=8)
    ax = fig.gca()
    
    ax.grid(False)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    plt.ylabel(r'Rate (day$^{-1}$)')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(indstr)
    ax.invert_xaxis()
    
    plt.title('Compartmental rates of tumor growth')
    
    plt.legend((p1[0], p2[0], p3[0]), (r'Birth ($k_{B,i}$)', r'Death ($k_{D,i}$)', r'Net growth ($k_{G,i}$)'))
    fig.tight_layout()
    
    plt.savefig('spatial_figs/CompRate.png')
    plt.show()


def plot_model_perf(subfig, t, matValid, maxIter, pname1, pval1, pname2, pval2, titleStr):

    plt.rcParams["legend.loc"] = 'best'
    
    # Latex conversion - p1
    if pname1 == 'kv':
        printName = r'$k_V$'
    elif pname == 'f':
        printName = r'$f_V$'
    elif pname == 't':
        printName = r'$t_{1/2}$'
    elif pname == 'O2max':
        printName = r'$C_0$'
    else:
        printName = pname

    # Latex conversion - p2
    if pname2 == 'kv':
        printName2 = r'$k_V$'
    elif pname2 == 'f':
        printName2 = 'r$f_V$'
    elif pname2 == 't':
        printName2 = r'$t_{1/2}$'
    elif pname2 == 'O2max':
        printName2 = r'$C_0$'
    else:
        printName2 = pname2

    Y = matValid.T
    X = np.matlib.repmat(t, Y.shape[1], 1).T

    plt.style.use('grayscale')

    font = {'family': 'normal','weight': 'normal','size': 17}
    plt.rc('font', **font)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

    colors = ["#000000", "#4a4a4a", "#4a4a4a", "#7a7a7a", "#7a7a7a"]
    markers = [10, "", "", "", "o"]
    starts = [0, 0, 0, 0, 300]
    ls = ["-", "--", "-.", ":", "-"]

    plt.title(titleStr)
    plt.xlabel('Time elapsed post-carcinogenesis (year)')
    
    labels = ["%.3f, %d" % (i,j) for i, j in zip(pval1, pval2)] + ["Hori et al"]
    legend_title = "parameters: " + printName + ", " + printName2
    
    ax = fig.gca()
    ax.grid(False)

    if subfig == 1 or subfig == 2:
        # do legend input
        for i in range(X.shape[1]):
            plt.semilogy(X[:, i], Y[:, i], linewidth=3, markersize=8, linestyle=ls[i],
                         marker=markers[i], markevery=(starts[i], 600), color=colors[i])
        plt.axis((0, t[-1], 1, 5 * 10**8))
        plt.ylabel(r'Population (log$_{10}$ cells)')
    
    if subfig == 1:
        # compare growth with Hori model
        Phori, qhori = calc_hori(maxIter)
        plt.semilogy(X[:, 1], Phori, 'k-', linewidth=3)
        plt.axis((0, t[-1], 1, 5 * 10**8))

    elif subfig == 3:
        for i in range(X.shape[1]):
            plt.plot(X[:, i], Y[:, i], linewidth=3, markersize=8, linestyle=ls[i],
                     marker=markers[i], markevery=(starts[i], 600), color=colors[i])
        # do legend input
        plt.axis((0, t[-1], np.min(Y[1, :]), np.max(Y[-1, :])))
        plt.ylabel(r'Necrotic fraction')
    
    plt.legend(labels, title=legend_title)
    plt.savefig('spatial_figs/lines_%s.png' % titleStr)
    plt.show()
    
    
#------------------
# Helper functions
#------------------

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)
    
def calc_hori(maxIter):
    """
    Hori et al's trajectory for plotting comparison.
    """
    # running the Hori SOLUTION model for parameter estimation:
    # k_GR = ln(2)/t_DT, where t_DT (tumor doubling time) = 120 days
    kGR = 5.78e-3
    N_T0 = 1
    
    # baseline values in Hori Model
    f = 0.1
    R = 4.5e-5
    th = 6.4 # half life in days
    kE = np.log(2) / th
    fRN_H = 4.56e3
    q_0 = 0 # assume basal of 0 for all
    dt = 1 # delta t

    N_soln = lambda t, N_T0, kGR: N_T0 * np.exp(kGR * t)
    q_soln = lambda t, kE, N, q: dt * (f*R*N[t-1]  + fRN_H - kE*q[t-1]) + q[t-1]
    
    N_vec = np.zeros((maxIter))
    N_vec[0] = N_T0
    q_vec = np.zeros((maxIter))
    q_vec[0] = q_0

    for t in range(0, maxIter): 
        N_vec[t] = N_soln(t, N_T0, kGR)
    
    for t in range(1, maxIter):
        q_vec[t] = q_soln(t, kE, N_vec, q_vec)
        
    return N_vec, q_vec


def grayscale_cmap(cmap):
    """
    Return a grayscale version of the given colormap
    """
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance - cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    

def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
