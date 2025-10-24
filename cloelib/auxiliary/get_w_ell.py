

def get_W_ell(self, ROOTS, norms, thetagrid, ns, ells):
        """
        Precomputes the kernel functions needed to calculate the COSEBI's

        Parameters:
        ROOTS (list): list containing the roots of the cosebi kernel functions, ROOTS[n] should have n + 1 entries
        norms (list): lit containing the the norms, should start at n[1]
        thetagrid (np.array): sufficiently dense grid going from thetamin to thetamax
        ns (list): the indices for the kernel function
        ells (np.array): array with the ells 

        Returns:
        dict with the kernel functions for each n
        """

        w_ell_dict = {}
        print('Calculating the kernel functions, this might take a couple of minutes.')
        for n in ns:
            # Calculate Tn+
            z = np.log(thetagrid / thetagrid[0])
            prod = 1
            for root in ROOTS[n-1]:
                prod *= (z - root)
            tp =  (norms[n] * prod)
            binsize = np.log(thetagrid[-1]/thetagrid[0])/(len(thetagrid)-1)
            # Do the integral to obtain Wn(ell)
            w_ell = np.zeros_like(ells)
            for i,ell in enumerate(ells):
                J0 = special.jv(0, ell * thetagrid)
                integrand = tp * thetagrid * J0 
                w_ell = w_ell.at[i].set(np.sum(integrand * thetagrid * binsize))
            
            w_ell_dict[n] = w_ell
        
        return w_ell_dict   