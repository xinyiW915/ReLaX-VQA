import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
def curve_bounds(x, params, sigma):
    upper_bound = logistic_func(x, params[0] + 2 * sigma[0], params[1] + 2 * sigma[1], params[2] + 2 * sigma[2], params[3] + 2 * sigma[3])
    lower_bound = logistic_func(x, params[0] - 2 * sigma[0], params[1] - 2 * sigma[1], params[2] - 2 * sigma[2], params[3] + 2 * sigma[3])
    return upper_bound, lower_bound

# plot one
def plot_results(y_test, y_test_pred_logistic, df_pred_score, network_name, model_name, data_name, layer_name, select_criteria):
    # nonlinear logistic fitted curve / logistic regression
    mos = y_test
    y = y_test_pred_logistic
    try:
        beta = [np.max(mos), np.min(mos), np.mean(y), 0.5]
        popt, pcov = curve_fit(logistic_func, y, mos, p0=beta, maxfev=100000000)
        sigma = np.sqrt(np.diag(pcov))
    except:
        raise Exception('Fitting logistic function time-out!!')
    x_values = np.linspace(np.min(y), np.max(y), len(y))

    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(10, 8))

    plt.plot(x_values, logistic_func(x_values, *popt), '-', color='#c72e29', label='Fitted f(x)')
    fig1 = sns.scatterplot(x="y_test_pred_logistic", y="MOS", data=df_pred_score, markers='o', color='steelblue', label=network_name, s=100)

    # set the legend to a location outside the plot and specify the bbox_to_anchor
    plt.legend(loc='lower right', fontsize=24, bbox_to_anchor=(1.0, 0.0))
    plt.ylim(1, 5)
    plt.xlim(1, 5)

    title_name = f"Algorithm {network_name} with {model_name} on dataset {data_name}: {select_criteria}"
    plt.title(title_name, fontsize=20)
    plt.xlabel('Predicted Score', fontsize=24)
    plt.ylabel('MOS', fontsize=24)
    reg_fig1 = fig1.get_figure()

    # save the file
    # fig_path = f'../../figs/{data_name}/'
    # if not os.path.exists(fig_path):
    #     os.makedirs(fig_path)
    # fig_name = f"{network_name}_{layer_name}_{model_name}_{data_name}_by{select_criteria}.png"
    # reg_fig1.savefig(f'{fig_path}{fig_name}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

# plot comparison
def plot_comparison(df1, df2, network_name, model_name, data_name, layer_name, compare1, compare2):
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x="y_test_pred_logistic", y="MOS", data=df1, ax=ax, palette='colorblind', marker='o', s=100, label=compare1)
    sns.scatterplot(x="y_test_pred_logistic", y="MOS", data=df2, ax=ax, palette='colorblind', marker='x', s=100, label=compare2)

    # adjust marker edge width for each scatter plot
    for scatter_plot in ax.collections:
        scatter_plot.set_linewidth(1.5)

    plt.rcParams.update({'font.size': 24})
    plt.legend(loc='lower right', title="Comparison", fontsize=24, bbox_to_anchor=(1.0, 0.0))

    plt.ylim(1, 5)
    plt.xlim(1, 5)

    title_name = f"Algorithm {network_name} with {model_name} on dataset {data_name}"
    plt.title(title_name, fontsize=24)
    plt.xlabel('Predicted Score', fontsize=24)
    plt.ylabel('MOS', fontsize=24)
    reg_fig = ax.get_figure()

    # save the file
    # fig_path = f'../../figs/{data_name}/'
    # fig_name = f"{network_name}_{layer_name}_{model_name}_{data_name}.png"
    # reg_fig.savefig(f'{fig_path}{fig_name}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
