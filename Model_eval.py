import numpy as np
import pandas as pd
from utils import *

dataframe = pd.read_csv('2 LearningAlgorithms/SAC/results/SAC_eval_result-20240830-14:28.csv')

plot_sequence(sequences=[
                         np.log2(dataframe['SNR_E'][:25]),
                         np.log2(dataframe['SNR_D'][:25] - 1),
              ],
              labels=[
                  r"$\gamma_\mathrm{E}$",
                  r"$\gamma_\mathrm{D}$",
              ],
              xlabel=r"$t$",
              ylabel=r"SNR",
              filename="SNR.pdf")

plot_sequence(sequences=[
                dataframe['n_E'][:25],
                dataframe['P_S'][:25]
              ],
              labels=[
                  r"$n_\mathrm{E}$",
                  r"$P_\mathrm{S}$,"
              ],
              xlabel=r"$t$",
              ylabel=r"Power",
              filename="Power.pdf")