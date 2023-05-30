# -*- coding: utf-8 -*-

# Import pyAgrum base module
import pyAgrum as gum
# Fonctionnalités de visualisation dans un Notebook
import pyAgrum.lib.notebook as gnb 


# BN init
bn = gum.BayesNet('Ex1')

# List of variables
var_names = ["E", "L", "A", "B"]
# BN variables creation loop
for var in var_names:
    # Create binary variable
    bn_var = gum.LabelizedVariable(var, '', 2)
    # Change labels
    [bn_var.changeLabel(i, lab) for i, lab in enumerate(["n", "o"])]
    # Add variable in the BN
    bn.add(bn_var)

# Add dependencies
bn.addArc("E", "L")
bn.addArc("L", "A")
bn.addArc("L", "B")

# Set CPTs
bn.cpt("E")[:] = [0.5, 0.5]

bn.cpt("L")[{'E': "n"}] = [1/10, 9/10]
bn.cpt("L")[{'E': "o"}] = [4/5, 1/5]

bn.cpt("A")[{'L': "n"}] = [3/4, 1/4]
bn.cpt("A")[{'L': "o"}] = [1/4, 3/4]

bn.cpt("B")[{'L': "n"}] = [9/10, 1/10]
bn.cpt("B")[{'L': "o"}] = [1/10, 9/10]

# Compute P(A) with variable elimination step by step
# ----------------------------------------------------
# Summing out B
psi_B = bn.cpt("B").margSumOut(["B"]) # = [1,1]
print(psi_B)

# Summing out E
phi_E = bn.cpt("L")*bn.cpt("E") # = P(L,E)
print(phi_E)
psi_E = phi_E.margSumOut(["E"]) # = P(L)
print(psi_E)

# Summing out L
phi_L = bn.cpt("A")*psi_E # = P(A,L)
print(phi_L)
psi_L = phi_L.margSumOut(["L"]) # = P(A)
print(psi_L)


# Check with inference engine
inf_eng = gum.LazyPropagation(bn)
inf_eng.makeInference()

print(inf_eng.posterior("A"))


# Compute P(L|B=n) with variable elimination step by step
# -------------------------------------------------------

# Summing out A is straightforward

# psi_E has already been previously computed

# P(L, B)
P_L_B = bn.cpt("B")*psi_E
print(P_L_B)

# Compute P(B=n)
P_B_n = P_L_B.margSumOut(["L"])[{"B": "n"}]

# Compute P(L| B=n)
P_L_c_B_n = P_L_B[{"B": "n"}]/P_B_n
print(P_L_c_B_n)

# Check with inference engine
inf_eng = gum.LazyPropagation(bn)
inf_eng.setEvidence({"B": "n"})
inf_eng.makeInference()

print(inf_eng.posterior("L"))


# Compute P(A,B|L=o) with variable elimination step by step
# ---------------------------------------------------------

# P(A,B,L)
P_A_B_L = bn.cpt("A")*bn.cpt("B")*psi_E

# Compute P(L=o)
P_L_o = P_A_B_L.margSumOut(["A", "B"])[{"L": "o"}]

# Compute P(A,B | L=o)
P_A_B_c_L_o = P_A_B_L[{"L": "o"}]/P_L_o
print(P_A_B_c_L_o)

# Check with inference engine
inf_eng = gum.LazyPropagation(bn)
inf_eng.addJointTarget({"A", "B"})
inf_eng.setEvidence({"L": "o"})
inf_eng.makeInference()

print(inf_eng.jointPosterior({"A", "B"}))
