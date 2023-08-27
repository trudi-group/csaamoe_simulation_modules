import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import yaml
import importlib
import sys
import pandas as pd
from matplotlib.legend import _get_legend_handles_labels
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import numpy as np

matplotlib.use("TkAgg")
pd.options.mode.chained_assignment = None  # default='warn'
path_settings = "./SETTINGS.yml"
path_gen_helpers = "./csaamoe_simulation_modules/gen_helpers.py"

# Import settings
with open(path_settings, 'r') as stream:
    try:
        SETTINGS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Import general helpers
spec = importlib.util.spec_from_file_location("noname", path_gen_helpers)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

# Import own modules
sys.path.append('./modules/')
import pricemodel
import re
from datetime import datetime


def give_weights_for_param(
     idx_t: list
    , param_fixed
    , param_variable_from: float
    , param_variable_to: float
    , param_variable_steps: float
    , type: str
):

    variable_params = pd.Series(np.linspace(param_variable_from
                                           , param_variable_to
                                           , param_variable_steps))
    idx_sum_end = (idx_t-1)
    out = pd.DataFrame([idx for idx in range(1, idx_sum_end)])
    for vp in variable_params:
        weights = list()
        ts = list()
        for idx in range(1, idx_sum_end):
            # ----- Setting vars -----
            # Components of first factor of sum
            how_far_in_past = idx_sum_end - (idx + 1) # "+1" as 0-indexing in python
            if type == "memory_varies":
                amplitude = param_fixed
                memory = vp
                weight = np.exp(-how_far_in_past / memory) * amplitude/memory
            elif type == "amplitude_varies":
                amplitude = vp
                memory = param_fixed
                weight = np.exp(-how_far_in_past / memory) * amplitude/memory
            else:
                raise Exception("Wrong input.")
            weights.append(weight)
            ts.append(idx)
        weights_proc = pd.DataFrame([w * 100 for w in weights])
        out = pd.concat([out, weights_proc], axis=1)
    # Rename Cols
    cols = ["t"]
    cols.extend(str(round(vp,1)) for vp in variable_params)
    out.columns = cols
    # Sort
    out = out.sort_values("t", ascending=True).reset_index(drop=True)

    return out


def make_k_plot(
        zetas_m_from
        , zetas_v_from
        , zetas_m_to
        , zetas_v_to
        , steps
        , amplitude_param_m
        , amplitude_param_v
        , memory_param_m
        , memory_param_v
        , type_k
        , path
):

    # Parameter space
    zetas_m = [i for i in np.linspace(zetas_m_from, zetas_m_to, steps)]
    zetas_v = [i for i in np.linspace(zetas_v_from, zetas_v_to, steps)]
    X, Y = np.meshgrid(zetas_m, zetas_v)

    # K-function
    Z = k_basic(amplitude_param_m=amplitude_param_m,
                amplitude_param_v=amplitude_param_v,
                memory_param_m=memory_param_m,
                memory_param_v=memory_param_v,
                zeta_m=X,
                zeta_v=Y,
                type=type_k)
    Z[Z > 1] = 1
    Z[Z < 0] = 0
    # Quadrant divider
    #Z_divider = X*0+Y*0+0.5

    # Assemble Plot

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color="black")
    #ax.plot_wireframe(X, Y, Z_divider, rstride=20, cstride=20, color="grey")

    # Plot customization
    ax.set_xlabel(r'$\zeta^{m}_{t-1}$')
    ax.set_ylabel(r'$\zeta^{v}_{t-1}$')
    ax.set_zlabel('$k_{t}$')
    ax.set_xlim3d(zetas_m_from, zetas_m_to)
    ax.set_ylim3d(zetas_v_from, zetas_v_to)
    ax.set_zlim3d(0.1, 1.1)

    # Save
    fig.set_size_inches(4, 4)
    fig.savefig(path+'k_variation.png', dpi=1600)


def k_basic(amplitude_param_v,
            amplitude_param_m,
            memory_param_m,
            memory_param_v,
            zeta_m,
            zeta_v,
            type):
    if type == "linear":
        k = 0.5 + \
            0.5 * (amplitude_param_m / memory_param_m) * zeta_m + \
            0.5 * (amplitude_param_v / memory_param_v) * zeta_v
    elif type == "tanh":
        k = 0.5 + \
            0.5 * np.tanh((amplitude_param_m / memory_param_m) * zeta_m + (amplitude_param_v / memory_param_v) * zeta_v)
    else:
        raise Exception("Wrong input.")
    return k


def plot_weight_shifts(path, memory_params, amplitude_params, idx_t):
    idx_t = idx_t
    memory_params   = memory_params#[2,7]
    amplitude_params = amplitude_params#[0.5,1]
    idx_sum_end = (idx_t - 1)
    weight_collection = list()
    column_names = list()
    for memory in memory_params:
        for amplitude in amplitude_params:
            column_names.append("weight_mem_{}_amp_{}".format(str(memory),str(amplitude)))
            weights = list()
            tau = list()
            for idx in range(1, idx_sum_end):
                # ----- Setting vars -----
                # Components of first factor of sum
                how_far_in_past = idx_sum_end - (idx + 1)  # "+1" as 0-indexing in python
                weight = np.exp(-how_far_in_past / memory) * amplitude / memory * 100
                weights.append(weight)
                tau.append(how_far_in_past)
            weights_proc = pd.DataFrame([w * 100 for w in weights])
            weight_collection.append(weights)

    # Form dataframe and rename columns
    weight_collection.append(tau)
    column_names.append("tau")
    df = pd.DataFrame(weight_collection).transpose()
    df.columns = column_names
    df = df.sort_values(by='tau', ascending=True)
    labels=[r'$c={},q={}$'.format(s.split("_")[2],s.split("_")[4]) for s in df.columns[:4]]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.plot(  'tau'
    , df.columns[0]
    , data=df
    , marker='o'
    , markerfacecolor='grey'
    , markersize=4
    , color='grey'
    , linewidth=1
    , linestyle='dashed'
    , label=labels[0])
    plt.plot(  'tau'
    , df.columns[1]
    , data=df
    , marker='o'
    , markerfacecolor='black'
    , markersize=4
    , color='black'
    , linewidth=1
    , linestyle='dashed'
    , label=labels[1])
    plt.plot(  'tau'
    , df.columns[2]
    , data=df
    , marker='o'
    , markerfacecolor='grey'
    , markersize=4
    , color='grey'
    , linewidth=1
    , linestyle='solid'
    , label=labels[2])
    plt.plot(  'tau'
    , df.columns[3]
    , data=df
    , marker='o'
    , markerfacecolor='black'
    , markersize=4
    , color='black'
    , linewidth=1
    , linestyle='solid'
    , label=labels[3])

    plt.legend()
    plt.gca().invert_xaxis()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'Weight $\frac{q}{c} \cdot e^{\frac{-(t-1-\tau)}{c}}$ in %')
    fig.set_size_inches(8, 5)
    fig.savefig(path+'weights_variation.png', dpi=1600)


def make_weighting_figure(
        idx_t
        , type
        , param_fixed
        , param_variable_from
        , param_variable_to
        , param_variable_steps
):
    df = give_weights_for_param(
        idx_t=idx_t
        , type=type
        , param_fixed=param_fixed
        , param_variable_from=param_variable_from
        , param_variable_to=param_variable_to
        , param_variable_steps=param_variable_steps
    )
    # Set up SNS
    sns.set()
    sns.set_palette("Set1", 10, .75)
    #plt.figure()
    # Prepare data
    melt = pd.melt(df.sort_values("t", ascending=True).reset_index(drop=True), id_vars=["t"])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = sns.lineplot(data=melt, x="t"
                      , y="value"
                      ,  hue="variable"
                      , marker="o"
                      #, dashes=False
                      , palette=sns.color_palette("Set1", melt.variable.nunique()))
    legend = ax.legend()
    legend.texts[0].set_text("Memory Parameters")
    # Customise some display properties
    ax.set_title('Weights Summands form price as calculated for t = {}'.format(str(round(idx_t))))
    ax.grid(color='#cccccc')
    ax.set_xlabel("t")
    ax.set_ylabel('Weights in % ')
    # Ask Matplotlib to show it
    plt.show()
    #fig.set_size_inches(5, 5)
    #fig.savefig(path+'weights_variation.png', dpi=1600)


def create_example_data(params):
    df_baseline = {
        "time": pd.date_range(datetime.today(), periods=params["n"]).tolist()
        , "fv": [params["fv_iter_start"]] * params["n"]
        , "m_total": [params["m_total_baseline"]] * params["n"]
        , "onchain_vol_usd": [params["onchain_vol_usd_baseline"]] * params["n"]
        , "v_circ": [params["v_circ_baseline"]] * params["n"]
    }
    df_baseline = pd.DataFrame(df_baseline)
    df = df_baseline.copy()
    return df


def load_simulation_data(params, dta, fv_col):
    df_baseline = {
        "time": pd.to_datetime(dta.time)
        , "fv": dta[fv_col]
        , "m_total": dta.m_total
        , "onchain_vol_usd": dta.onchain_vol_usd
        , "v_circ": dta.velocity
        , "prices_observed": dta.prices
    }
    df_baseline = pd.DataFrame(df_baseline)
    df = df_baseline.copy()
    df = df.head(params["n"])
    return df


def simulation(params, simulation_input):
    # -- get simulation results
    colnames_prices = []
    colnames_component = []
    simulation_results_prices = list()
    simulation_results_components = list()
    for c_v in params["memory_param_v"]:
        for c_m in params["memory_param_m"]:
            for q_v in params["amplitude_param_v"]:
                for q_m in params["amplitude_param_m"]:
                    pack_1 = pricemodel.get_theoretic_prices(
                        input_dta=simulation_input
                        , memory_param_m=c_m
                        , memory_param_v=c_v
                        , amplitude_param_m=q_m
                        , amplitude_param_v=q_v
                        , induction_start_prices_1=params["prices_iter_start_1"]
                        , induction_start_prices_2=params["prices_iter_start_2"]
                        , idx_start=2
                        , precision=params["precision"]
                        , reset_logs=True
                        , dir_logging=None
                        , verbose=False
                        , truncate_zH_until_period=params["truncate_zH_until_period"]
                    ).copy()
                    pack_2 = pricemodel.get_components(
                        input_dta=simulation_input
                        , memory_param_m=c_m
                        , memory_param_v=c_v
                        , amplitude_param_m=q_m
                        , amplitude_param_v=q_v
                        , precision=params["precision"]
                        , reset_logs=True
                        , dir_logging=None
                        , verbose=False
                        , truncate_zH_until_period=params["truncate_zH_until_period"]
                    ).copy()
                    simulation_results_prices.append(pack_1.prices)
                    simulation_results_components.append(pack_2.m_hodl)
                    id_string = "_qv" + str(q_v) + "_qm" + str(q_m) + "_cv" + str(c_v) + "_cm" + str(c_m)
                    colnames_prices.append("prices" + id_string)
                    colnames_component.append("hodl" + id_string)
    # list of lists to dataframe
    simulation_results_prices = pd.DataFrame(simulation_results_prices)
    simulation_results_components = pd.DataFrame(simulation_results_components)
    simulation_results_prices = simulation_results_prices.transpose()
    simulation_results_components = simulation_results_components.transpose()
    simulation_results_prices.columns = colnames_prices
    simulation_results_components.columns = colnames_component

    out = {"prices": simulation_results_prices,
           "components": simulation_results_components}
    return out


def simulation_simple(params, simulation_input):
    # -- get simulation results
    colnames_prices = []
    colnames_component = []
    simulation_results_prices = list()
    simulation_results_components = list()
    for p in params["amplitude_param_m"]:
        pack_1 = pricemodel.get_theoretic_prices(
            input_dta=simulation_input
            , memory_param_m=params["memory_param_m"]
            , memory_param_v=params["memory_param_v"]
            , amplitude_param_m=p
            , amplitude_param_v=params["amplitude_param_v"]
            , induction_start_prices_1=params["prices_iter_start_1"]
            , induction_start_prices_2=params["prices_iter_start_2"]
            , idx_start=2
            , precision=params["precision"]
            , reset_logs=True
            , dir_logging=None
            , verbose=False
            , truncate_zH_until_period=params["truncate_zH_until_period"]
        ).copy()
        pack_2 = pricemodel.get_components(
                input_dta=simulation_input
                , memory_param_m=params["memory_param_m"]
                , memory_param_v=params["memory_param_v"]
                , amplitude_param_m=p
                , amplitude_param_v=params["amplitude_param_v"]
                , precision=params["precision"]
                , reset_logs=True
                , dir_logging=None
                , verbose=False
                , truncate_zH_until_period=params["truncate_zH_until_period"]
            ).copy()

        simulation_results_prices.append(pack_1.prices)
        simulation_results_components.append(pack_2.m_hodl)
        colnames_prices.append("prices_" + str(p))
        colnames_component.append("hodl_" + str(p))
    # list of lists to dataframe
    simulation_results_prices = pd.DataFrame(simulation_results_prices)
    simulation_results_components = pd.DataFrame(simulation_results_components)
    simulation_results_prices = simulation_results_prices.transpose()
    simulation_results_components = simulation_results_components.transpose()
    simulation_results_prices.columns = colnames_prices
    simulation_results_components.columns = colnames_component

    out = {"prices": simulation_results_prices,
           "components": simulation_results_components}
    return out


def plotting_guts_mechanics(params, simulation_output, simulation_input):

    # -- Prepare simulation results for plotting
    basic_variables = {
      't': [i for i in range(1, len(simulation_input) + 1)]
    , 'm_total': simulation_input.m_total
    , 'cash_balance': simulation_input.onchain_vol_usd/simulation_input.v_circ
    , 'fv': simulation_input.fv
            }
    basic_variables = pd.DataFrame(basic_variables)
    basic_variables_ordered = basic_variables[['t', 'm_total', 'cash_balance', 'fv']]
    plotting_data = pd.concat([basic_variables_ordered
                                  , simulation_output["components"]
                                  , simulation_output["prices"]], axis=1, sort=False)

    # -- Gather inputs for plotting
    # - labels
    labels = [r"$M_t$", r"$B_t$", r"$F_{t}$"]

    idx_begin = len(basic_variables_ordered.columns)
    idx_end = len(basic_variables_ordered.columns)+len(simulation_output["prices"].columns)
    labels_auto_prices = [r'$S_t$ with $q_m={}$'.format(s.split("_")[1]) for s in plotting_data.columns[idx_begin:idx_end]]

    idx_begin = len(basic_variables_ordered.columns)+len(simulation_output["prices"].columns)
    idx_end = len(basic_variables_ordered.columns) + len(simulation_output["prices"].columns) + len(simulation_output["components"].columns)
    labels_auto_component = [r'$Z_t$ with $q_m={}$'.format(s.split("_")[1]) for s in plotting_data.columns[4:]]

    labels.extend(labels_auto_component)
    labels.extend(labels_auto_prices)

    line_styles = ["solid", "solid", "solid", "solid", "dashed" , "solid", "dotted","solid", "solid" , "dashed", "dotted", "dashdot"]
    line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    colors = ["black", "firebrick", "green", "blue", "dimgrey", "dimgrey", "dimgrey",  "blue", "dimgrey", "dimgrey", "dimgrey","dimgrey"]

    # - assemble plot
    fig, (ax_basic_vars, ax_simulated_prices) = plt.subplots(nrows=params["plot_rows"], ncols=params["plot_cols"])

    # - plot price simulations
    ax_basic_vars.plot('t'
             , plotting_data.columns[3]
             , data=plotting_data
             , marker=None
             , color=colors[2]
             , linewidth=line_widths[2]
             , linestyle=line_styles[2]
             , label=labels[2])
    ax_basic_vars_second_axis = ax_basic_vars.twinx()
    ax_basic_vars_second_axis.plot('t'
             , plotting_data.columns[1]
             , data=plotting_data
             , marker=None
             , color=colors[0]
             , linewidth=line_widths[0]
             , linestyle=line_styles[0]
             , label=labels[0])
    ax_basic_vars_second_axis.plot('t'
             , plotting_data.columns[2]
             , data=plotting_data
             , marker=None
             , color=colors[1]
             , linewidth=line_widths[1]
             , linestyle=line_styles[1]
             , label=labels[1])
    for p in range(4, 8):
        ax_basic_vars_second_axis.plot('t'
                 , plotting_data.columns[p]
                 , data=plotting_data
                 , marker=None
                 , color=colors[p - 1]
                 , linewidth=line_widths[p - 1]
                 , linestyle=line_styles[p - 1]
                 , label=labels[p - 1])

    ax_basic_vars.legend(*_get_legend_handles_labels(fig.axes), loc='upper center', bbox_to_anchor=(0.5, -0.125),
                     fancybox=False, shadow=False, ncol=1, frameon=False) # needs to stay here
    for p in range(8, len(plotting_data.columns)):
        ax_simulated_prices.plot('t'
                 , plotting_data.columns[p]
                 , data=plotting_data
                 , marker=None
                 , color=colors[p - 1]
                 , linewidth=line_widths[p - 1]
                 , linestyle=line_styles[p - 1]
                 , label=labels[p - 1])
    ax_simulated_prices.legend(*_get_legend_handles_labels([fig.axes[1]]), loc='upper center', bbox_to_anchor=(0.5, -0.125),
                     fancybox=False, shadow=False, ncol=1, frameon=False)

    # - Y-axis limits
    ax_simulated_prices.set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    ax_basic_vars.set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    ax_basic_vars_second_axis.set_ylim(params["ylim_modelvars_from"], params["ylim_modelvars_to"])

    # - axis labels
    ax_simulated_prices.set_xlabel(r'$t$')
    ax_basic_vars.set_xlabel(r'$t$')
    ax_simulated_prices.set_ylabel(r'$S_t$ in $\frac{USD}{CC}$')
    ax_basic_vars.set_ylabel(r'$F_{t}$ in $CC$')
    ax_basic_vars_second_axis.set_ylabel(r'$B_t$ in USD; $M_t$,$Z_t$ in $CC$')

    # - titles
    ax_basic_vars.set_title(r"Panel (1): Variables constituting QTM ", fontsize=10)
    ax_simulated_prices.set_title(r"Panel (2): Simulated price processes", fontsize=10)

    # - general plot settings
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(wspace=1, bottom=params["fig_space_bottom"])

    return fig


def strip_packed_parameters(s):
    packed_parameters = s.split("_")
    param_list = ["qv", "qm", "cv", "cm"]  # Order matters!
    packed_parameters_of_interest = [i for i in packed_parameters if any(j in i for j in param_list)]
    parameter_values = [re.sub("[A-Za-z]", "", i) for i in packed_parameters_of_interest]
    out = dict(zip(["qv", "qm", "cv", "cm"], parameter_values))
    return out

def strip_column_names(cols, variable_id, additional_restrictions):
    cols_of_interest = [i for i in cols if variable_id in i]
    if additional_restrictions:
        for r in additional_restrictions:
            cols_of_interest = [i for i in cols_of_interest if r in i]
    return cols_of_interest


def mechanics_param_illustrations(params, simulation_output, simulation_input):
    # -- Prepare simulation results for plotting
    basic_variables = {
        't': [i for i in range(1, len(simulation_input) + 1)]
        , 'm_total': simulation_input.m_total
        , 'cash_balance': simulation_input.onchain_vol_usd / simulation_input.v_circ
        , 'fv': simulation_input.fv
    }
    basic_variables = pd.DataFrame(basic_variables)
    basic_variables_ordered = basic_variables[['t', 'm_total', 'cash_balance', 'fv']]
    plotting_data = pd.concat([basic_variables_ordered
                                  , simulation_output["components"]
                                  , simulation_output["prices"]], axis=1, sort=False)

    # -- Gather inputs for plotting
    # - data for subplots
    basic_variables = ["fv"]
    plot_1_cols_of_interest = strip_column_names(plotting_data.columns, variable_id="prices",
                                                 additional_restrictions=["qm0", "cv40", "cm40"])
    plot_2_cols_of_interest = strip_column_names(plotting_data.columns, variable_id="prices",
                                                 additional_restrictions=["qv40", "cv40", "cm40"])
    plot_3_cols_of_interest = strip_column_names(plotting_data.columns, variable_id="prices",
                                                 additional_restrictions=["qv40", "qm0", "cm40"])
    plot_4_cols_of_interest = strip_column_names(plotting_data.columns, variable_id="prices",
                                                 additional_restrictions=["qm10", "qv40", "cv40"])

    # - some manual cleaning. we cannot plot everything.
    plot_1_cols_of_interest = basic_variables + list(plot_1_cols_of_interest[i] for i in [10, 7, 3, 0])
    plot_2_cols_of_interest = basic_variables + list(plot_2_cols_of_interest[i] for i in [8, 6, 4, 2])
    plot_3_cols_of_interest = basic_variables + list(plot_3_cols_of_interest[i] for i in [6, 5, 4, 3])
    plot_4_cols_of_interest = basic_variables + list(plot_4_cols_of_interest[i] for i in [6, 4, 2, 1])

    # - styling
    line_styles = ["solid", "solid", "dashdot", "dashed", "dotted"]
    line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    colors = ["green", "dimgrey", "dimgrey", "dimgrey", "dimgrey"]

    # - assemble plot
    fig, axs = plt.subplots(1,1,squeeze=False)#params["plot_rows"], params["plot_cols"])
    # - plot price simulations
    for p in range(len(plot_1_cols_of_interest)):
        axs[0, 0].plot('t'
                       , plot_1_cols_of_interest[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label="_nolegend_" if plot_1_cols_of_interest[p] == "fv" else r"$q_v={}$".format(
                strip_packed_parameters(plot_1_cols_of_interest[p])["qv"]))
    for p in range(len(plot_2_cols_of_interest)):
        axs[1, 0].plot('t'
                       , plot_2_cols_of_interest[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label="_nolegend_" if plot_2_cols_of_interest[p] == "fv" else r"$q_m={}$".format(
                strip_packed_parameters(plot_2_cols_of_interest[p])["qm"]))
    for p in range(len(plot_3_cols_of_interest)):
        axs[0, 1].plot('t'
                       , plot_3_cols_of_interest[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label="_nolegend_" if plot_3_cols_of_interest[p] == "fv" else r"$c_v={}$".format(
                strip_packed_parameters(plot_3_cols_of_interest[p])["cv"]))
    for p in range(len(plot_4_cols_of_interest)):
        axs[1, 1].plot('t'
                       , plot_4_cols_of_interest[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label="_nolegend_" if plot_4_cols_of_interest[p] == "fv" else r"$c_m={}$".format(
                strip_packed_parameters(plot_4_cols_of_interest[p])["cm"]))

    # - title
    # axs[0, 0].set_title(r"Panel (1): $q_m=0$,$c_m=40$,$c_v=40$", fontsize=10)
    # axs[0, 1].set_title(r"Panel (2): $q_v=40$, $c_m=40$,$c_v=40$", fontsize=10)
    # axs[1, 0].set_title(r"Panel (3): $q_m=0$, $q_v=40$, $c_m=40$", fontsize=10)
    # axs[1, 1].set_title(r"Panel (4): $q_m=10$, $q_v=40$, $c_v=40$", fontsize=10)
    axs[0, 0].set_title(r"Panel (1): $q_m=0$,$c_m=40$,$c_v=40$", fontsize=10)
    axs[0, 1].set_title(r"Panel (2): $q_m=0$, $q_v=40$, $c_m=40$", fontsize=10)
    axs[1, 0].set_title(r"Panel (3): $q_v=40$, $c_m=40$,$c_v=40$", fontsize=10)
    axs[1, 1].set_title(r"Panel (4): $q_m=10$, $q_v=40$, $c_v=40$", fontsize=10)

    # - legend
    axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    # - y-lim
    axs[0, 0].set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    axs[0, 1].set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    axs[1, 0].set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    axs[1, 1].set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])

    axs[0, 0].set_xlabel(r'$t$')
    axs[0, 1].set_xlabel(r'$t$')
    axs[1, 0].set_xlabel(r'$t$')
    axs[1, 1].set_xlabel(r'$t$')


    axs[0, 0].set_ylabel(r'$F_t,S_t$ in $\frac{USD}{CC}$')
    axs[0, 1].set_ylabel(r'$F_t,S_t$ in $\frac{USD}{CC}$')
    axs[1, 0].set_ylabel(r'$F_t,S_t$ in $\frac{USD}{CC}$')
    axs[1, 1].set_ylabel(r'$F_t,S_t$ in $\frac{USD}{CC}$')

    # - general figure scaling
    fig.set_size_inches(params["width"], params["height"])
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    return fig


def make_simulation_real_data(path, params, dta, fv_col):
    # -- create shocked dataframe
    simulation_input = load_simulation_data(params=params,dta=dta,fv_col=fv_col)

    # -- run simulation and create figure
    simulation_output = simulation_simple(params=params, simulation_input=simulation_input)
    fig = plotting_guts_mechanics_real_data(params=params, simulation_input=simulation_input, simulation_output=simulation_output)

    # -- save
    fig.savefig(path + 'simulation_real_hayes_'+fv_col+'.png', dpi=1600)


def plotting_guts_mechanics_real_data(params, simulation_input, simulation_output):

    # -- Prepare simulation results for plotting
    basic_variables = {
      't': [i for i in range(1, len(simulation_input) + 1)]
    , 'm_total': simulation_input.m_total
    , 'cash_balance': simulation_input.onchain_vol_usd/simulation_input.v_circ
    , 'fv': simulation_input.fv
    , 'prices_observed': simulation_input.prices_observed
            }
    basic_variables = pd.DataFrame(basic_variables)
    basic_variables_ordered = basic_variables[['t', 'm_total', 'cash_balance', 'fv', 'prices_observed']]
    plotting_data = pd.concat([basic_variables_ordered
                                  , simulation_output["components"]
                                  , simulation_output["prices"]], axis=1, sort=False)

    # -- Gather inputs for plotting
    # - labels
    labels = [r"$M_t$", r"$B_t$", r"$F_{t}$", r"$\tilde{S}_t$"]

    idx_begin = len(basic_variables_ordered.columns)
    idx_end = len(basic_variables_ordered.columns)+len(simulation_output["prices"].columns)
    labels_auto_prices = [r'$S_t$ with $q_m={}$'.format(s.split("_")[1]) for s in plotting_data.columns[idx_begin:idx_end]]

    idx_begin = len(basic_variables_ordered.columns)+len(simulation_output["prices"].columns)
    idx_end = len(basic_variables_ordered.columns) + len(simulation_output["prices"].columns) + len(simulation_output["components"].columns)
    labels_auto_component = [r'$Z_t$ with $q_m={}$'.format(s.split("_")[1]) for s in plotting_data.columns[idx_begin:idx_end]]

    labels.extend(labels_auto_component)
    labels.extend(labels_auto_prices)

    line_styles = ["dashdot", "dashed", "solid", "solid", "solid", "dashed" , "dashdot", "dotted","solid", "dashed" , "solid", "dotted"]
    line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    colors = ["black", "black", "blue", "red", "silver", "dimgrey", "dimgrey", "dimgrey",  "silver", "dimgrey", "dimgrey", "dimgrey"]

    # - assemble plot
    fig, (ax_basic_vars, ax_simulated_prices) = plt.subplots(nrows=params["plot_rows"], ncols=params["plot_cols"])

    # - plot price simulations
    # ax_basic_vars.plot('t'
    #          , plotting_data.columns[3]
    #          , data=plotting_data
    #          , marker=None
    #          , color=colors[2]
    #          , linewidth=line_widths[2]
    #          , linestyle=line_styles[2]
    #          , label=labels[2])
    ax_basic_vars_second_axis = ax_basic_vars.twinx()
    ax_basic_vars_second_axis.plot('t'
             , plotting_data.columns[1]
             , data=plotting_data
             , marker=None
             , color=colors[0]
             , linewidth=line_widths[0]
             , linestyle=line_styles[0]
             , label=labels[0])
    ax_basic_vars_second_axis.plot('t'
             , plotting_data.columns[2]
             , data=plotting_data
             , marker=None
             , color=colors[1]
             , linewidth=line_widths[1]
             , linestyle=line_styles[1]
             , label=labels[1])
    for p in range(8, 5):
        ax_basic_vars_second_axis.plot('t'
                 , plotting_data.columns[p]
                 , data=plotting_data
                 , marker=None
                 , color=colors[p - 1]
                 , linewidth=line_widths[p - 1]
                 , linestyle=line_styles[p - 1]
                 , label=labels[p - 1])

    ax_basic_vars.legend(*_get_legend_handles_labels(fig.axes), loc="upper left") # needs to stay here
    for p in list(range(8, len(plotting_data.columns)-1))+[3,4]:
        ax_simulated_prices.plot('t'
                 , plotting_data.columns[p]
                 , data=plotting_data
                 , marker=None
                 , color=colors[p - 1]
                 , linewidth=line_widths[p - 1]
                 , linestyle=line_styles[p - 1]
                 , label=labels[p - 1])

    ax_simulated_prices.legend(*_get_legend_handles_labels([fig.axes[1]]), loc="upper left")

    # ax_simulated_prices.set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    # ax_basic_vars.set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])
    # ax_basic_vars_second_axis.set_ylim(params["ylim_modelvars_from"], params["ylim_modelvars_to"])

    ax_simulated_prices.set_xlabel(r'$t$')
    ax_basic_vars.set_xlabel(r'$t$')
    ax_simulated_prices.set_ylabel(r'$S_t$')
    ax_basic_vars.set_ylabel(r'$F_{t}$')
    ax_basic_vars_second_axis.set_ylabel(r'$F_{t},M_t,Z_t$')

    fig.set_size_inches(8, 5)
    fig.subplots_adjust(wspace=1, bottom=0.35)

    return fig


def make_simulation_shift_fv(path, fname, params):

    # -- create shocked dataframe
    simulation_input = create_example_data(params)
    simulation_input["fv"][params["n_shock_1"]:params["n"]] = [params["level_shock_1"]] * (params["n"] - params["n_shock_1"])
    simulation_input["fv"][params["n_shock_2"]:params["n"]] = [params["level_shock_2"]] * (params["n"] - params["n_shock_2"])
    simulation_input["fv"][params["n_shock_3"]:params["n"]] = [params["level_shock_3"]] * (params["n"] - params["n_shock_3"])
    simulation_input["fv"][params["n_shock_4"]:params["n"]] = [params["level_shock_4"]] * (params["n"] - params["n_shock_4"])

    # -- run simulation and create figure
    simulation_output = simulation_simple(params=params, simulation_input=simulation_input)
    fig = plotting_guts_mechanics(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def make_simulation_temporary_shock_txvol(path, fname, params):

    # -- create shocked dataframe
    simulation_input = create_example_data(params)
    simulation_input["onchain_vol_usd"][params["n_shock"]:(params["n_shock"]+params["duration_shock"])] = [params["level_shock"]]*(params["duration_shock"])

    # -- run simulation and create figure
    simulation_output = simulation_simple(params=params, simulation_input=simulation_input)
    fig = plotting_guts_mechanics(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def make_simulation_increasing_fv(path, fname, params):

    # -- create shocked dataframe
    duration_shock_1 = params["n_shock_1_end"]-params["n_shock_1_start"]
    duration_shock_2 = params["n_shock_2_end"] - params["n_shock_2_start"]
    old_fv_level = params["fv_iter_start"]
    intermediate_fv_level = params["fv_iter_start"] + params["level_shock_1"] * len(range(0, duration_shock_1))
    new_fv_level = intermediate_fv_level - params["level_shock_2"] * len(range(0, duration_shock_2))

    simulation_input = create_example_data(params)
    # - shock 1
    simulation_input["fv"][params["n_shock_1_start"]:params["n_shock_1_end"]] = [
        old_fv_level + params["level_shock_1"] * i for i in range(0, duration_shock_1)]
    simulation_input["fv"][params["n_shock_1_end"]:] = intermediate_fv_level
    # - shock 2
    simulation_input["fv"][params["n_shock_2_start"]:params["n_shock_2_end"]] = [
        intermediate_fv_level - params["level_shock_2"] * i for i in range(0, duration_shock_2)]
    simulation_input["fv"][params["n_shock_2_end"]:] = new_fv_level

    # -- run simulation and create figure
    simulation_output = simulation_simple(params=params, simulation_input=simulation_input)
    fig = plotting_guts_mechanics(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def make_parameter_illustration(path, fname, params):

    # -- create shocked dataframe

    new_fv_level = params["fv_iter_start"] + params["level_shock"]
    old_fv_level = params["fv_iter_start"]
    simulation_input = create_example_data(params)
    simulation_input["fv"][params["n_shock"]:params["n"]] = [params["level_shock"]] * (params["n"] - params["n_shock"])

    # -- run simulation and create figure
    simulation_output = simulation(params=params, simulation_input=simulation_input)
    fig = mechanics_param_illustrations(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname +'.png', dpi=1600)


def make_speculation_level_and_prices(path, fname, params):

    # -- create shocked dataframe
    duration_shock_1 = params["n_shock_1_end"]-params["n_shock_1_start"]
    duration_shock_2 = params["n_shock_2_end"] - params["n_shock_2_start"]
    duration_shock_3 = params["n_shock_3_end"] - params["n_shock_3_start"]
    old_fv_level = params["fv_iter_start"]
    intermediate_fv_level_1 = old_fv_level + params["level_shock_1"] * len(range(0, duration_shock_1))
    intermediate_fv_level_2 = intermediate_fv_level_1 + params["level_shock_2"] * len(range(0, duration_shock_2))
    new_fv_level = intermediate_fv_level_2 + params["level_shock_3"] * len(range(0, duration_shock_3))

    simulation_input = create_example_data(params)
    # - shock 1
    simulation_input["fv"][params["n_shock_1_start"]:params["n_shock_1_end"]] = [
        old_fv_level + params["level_shock_1"] * i for i in range(0, duration_shock_1)]
    # - shock 2
    simulation_input["fv"][params["n_shock_2_start"]:params["n_shock_2_end"]] = [
        intermediate_fv_level_1 + params["level_shock_2"] * i for i in range(0, duration_shock_2)]
    # - shock 3
    simulation_input["fv"][params["n_shock_3_start"]:params["n_shock_3_end"]] = [
        intermediate_fv_level_2 + params["level_shock_3"] * i for i in range(0, duration_shock_3)]

    # -- run simulation and create figure
    simulation_output = simulation(params=params, simulation_input=simulation_input)
    fig = plotting_guts_speculation_level_and_prices(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def get_labels_from_variables(col_name, typ, typ_label):
    if col_name == "m_total":
        label = r"$M_{t}$"
    elif col_name == "cash_balance":
        label = r"$B_{t}$"
    elif col_name == "fv":
        label = r"$V_{t}$"
    else:
        label = typ_label+r"$={}$".format(strip_packed_parameters(col_name)[typ])
    return label


def plotting_guts_speculation_level_and_prices(params, simulation_output, simulation_input):
    # -- Prepare simulation results for plotting
    basic_variables = {
        't': [i for i in range(1, len(simulation_input) + 1)]
        , 'm_total': simulation_input.m_total
        , 'cash_balance': simulation_input.onchain_vol_usd / simulation_input.v_circ
        , 'fv': simulation_input.fv
    }
    basic_variables = pd.DataFrame(basic_variables)
    basic_variables_ordered = basic_variables[['t', 'm_total', 'cash_balance', 'fv']]
    plotting_data = pd.concat([basic_variables_ordered
                                  , simulation_output["components"]
                                  , simulation_output["prices"]], axis=1, sort=False)

    # -- Gather inputs for plotting
    # - data for subplots
    basic_variables_qtm = ['m_total', 'cash_balance']
    basic_variables_prices = ["fv"]

    plot_cols_of_interest_prices = strip_column_names(plotting_data.columns, variable_id="prices", additional_restrictions=[])
    plot_cols_of_interest_qtm = strip_column_names(plotting_data.columns, variable_id="hodl", additional_restrictions=[])

    # - some manual cleaning. we cannot plot everything.
    plot_cols_of_interest_qtm = basic_variables_qtm + plot_cols_of_interest_qtm
    plot_cols_of_interest_prices = basic_variables_prices + plot_cols_of_interest_prices

    # - styling
    line_styles_qtm = ["solid", "solid", "dashdot", "dashed", "dotted"]
    line_widths_qtm = [1, 1, 1, 1, 1]
    colors_qtm = ["black", "black", "dimgrey", "dimgrey", "dimgrey"]
    line_styles_prices = ["solid", "dashdot", "dashed", "dotted"]
    line_widths_prices = [1, 1, 1, 1]
    colors_prices = ["black", "dimgrey", "dimgrey", "dimgrey"]

    # - assemble plot
    fig, axs = plt.subplots(1, 2)
    # - plot price simulations
    for p in range(len(plot_cols_of_interest_qtm)):
        axs[0].plot('t'
                       , plot_cols_of_interest_qtm[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors_qtm[p]
                       , linewidth=line_widths_qtm[p]
                       , linestyle=line_styles_qtm[p]
                       , label=get_labels_from_variables(plot_cols_of_interest_qtm[p], typ="qv", typ_label=r"$q_v$")
                    )
    for p in range(len(plot_cols_of_interest_prices)):
        axs[1].plot('t'
                       , plot_cols_of_interest_prices[p]
                       , data=plotting_data
                       , marker=None
                       , color=colors_prices[p]
                       , linewidth=line_widths_prices[p]
                       , linestyle=line_styles_prices[p]
                       , label=get_labels_from_variables(plot_cols_of_interest_prices[p], typ="qv", typ_label=r"$q_v$")
                    )

    # - title
    axs[0].set_title(r"Panel (1): QTM-Variables", fontsize=10)
    axs[1].set_title(r"Panel (2): Price-Simulations", fontsize=10)

    # - legend
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[0].set_ylim(params["ylim_modelvars_from"], params["ylim_modelvars_to"])
    axs[1].set_ylim(params["ylim_prices_from"], params["ylim_prices_to"])

    # - general figure scaling
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(wspace=0.5, hspace=0.5, bottom=params["fig_space_bottom"])

    return plt


def find_convergence_value(vec, threshold):
    if vec.tail().var() < threshold:
        convergence_to = vec.tail(20).mean()
    else:
        convergence_to = np.nan
    return convergence_to


def check_monotony(vec, threshold):
    diffs_rounded = vec.diff().round(decimals=threshold).dropna()
    monotonous = all(diffs_rounded <= 0) or all(diffs_rounded >= 0) #(not any(diffs_rounded < 0)) or (not any(diffs_rounded < 0))
    return monotonous


def make_convergence_df(params
                        , typ
                        , variable):
    df_monotony = pd.DataFrame()
    df_gap = pd.DataFrame()
    df_convergence = pd.DataFrame()
    params_clone = params.copy()

    for shock_size in params["level_shock"]:

        params_clone["level_shock"] = shock_size

        if typ == "jump":
            new_fv_level = params_clone["level_shock"]
            old_fv_level = params_clone["fv_iter_start"]
            input = create_example_data(params_clone)
            input["fv"][params_clone["n_shock"]:params_clone["n"]] = [params_clone["level_shock"]] * (params_clone["n"] - params_clone["n_shock"])
            output = simulation(params=params_clone, simulation_input=input)

        elif typ == "linear_increase":
            new_fv_level = params_clone["fv_iter_start"] + params_clone["level_shock"] * len(range(0, params_clone["duration_shock"]))
            old_fv_level = params_clone["fv_iter_start"]
            input = create_example_data(params_clone)
            input["fv"][params_clone["n_shock"]:(params_clone["n_shock"]+params_clone["duration_shock"])] = [old_fv_level+params_clone["level_shock"]*i for i in range(0,params_clone["duration_shock"])]
            input["fv"][(params_clone["n_shock"]+params_clone["duration_shock"]):] = new_fv_level
            output = simulation(params=params_clone, simulation_input=input)

        convergence_to = [find_convergence_value(output[variable][col], threshold=params_clone["sd_threshold_convergence"]) for col in output[variable].columns]
        convergence_to = pd.DataFrame(convergence_to, index=list(output[variable].columns.values))

        gap_left = ((new_fv_level- convergence_to)/new_fv_level)*100

        monotony = [check_monotony(output[variable][col],threshold=params_clone["rounding_threshold_monotony"]) for col in output[variable].columns]
        monotony = pd.DataFrame(monotony, index=output[variable].columns)

        # -- gather results
        colname = str(new_fv_level)
        df_monotony.loc[:, colname] = monotony[0]
        df_gap.loc[:, colname] = gap_left[0]
        df_convergence.loc[:, colname] = convergence_to[0]

    return [df_convergence,df_monotony, df_gap]


def make_plots_parameter_convergence(
        variable
        , typ_shock
        , typ_plot
        , params
        , path
        , fname):

    df_packed = make_convergence_df(params=params
                        , typ=typ_shock
                        , variable=variable)

    df_convergence = df_packed[0]; df_monotony = df_packed[1]; df_gap = df_packed[2]

    if typ_plot == "gap":
        df = df_gap
    elif typ_plot == "convergence":
        df = df_convergence

    # -- Gather inputs for plotting
    # - data for subplots
    # -
    converging = {'qv': [float(strip_packed_parameters(i)["qv"]) for i in df.index.values]}
    converging = pd.DataFrame(converging)
    converging = pd.concat(
        [converging.reset_index(drop=True), df[df_monotony].reset_index(drop=True)], axis=1,
        sort=False)
    converging.columns = ["var_" + str(s) for s in converging.columns]
    # -
    non_converging = {'qv': [float(strip_packed_parameters(i)["qv"]) for i in df.index.values]}
    non_converging = pd.DataFrame(non_converging)
    non_converging = pd.concat(
        [non_converging.reset_index(drop=True), df[~df_monotony].reset_index(drop=True)], axis=1,
        sort=False)
    non_converging.columns = ["var_" + str(s) for s in non_converging.columns]

    # - styling
    legends = [float(i) for i in df.columns]
    marker_styles = ["o", "v", "s", "D", "X", ">", "<", "H", "d", "p", "^", "."]
    label_var = (r"F" if variable == "prices" else "Z")

    # - plot price simulations
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
    for p in range(0,len(legends)):
        ax[0,0].plot( "var_qv"
                    , converging.columns[p+1]
                    , data=converging
                    , marker=marker_styles[p]
                    , markerfacecolor = 'black'
                    , markeredgecolor = "black"
                    , label=params["line_lable_before_equality"]+"{}".format(legends[p])
                    , linestyle = "None"
                    )
    for p in range(0,len(legends)):
        ax[0,0].plot( "var_qv"
                    , non_converging.columns[p+1]
                    , data=non_converging
                    , marker=marker_styles[p]
                    , markerfacecolor = "None"
                    , markeredgecolor = "black"
                    , linestyle = "None"
                    , label="_no_label_"
                    )

    # - title
    ax[0,0].set_title(r"", fontsize=10)

    # - legend
    ax[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)

    # - axis labels
    ax[0,0].set_xlabel(r'$q_{v}$')
    ax[0,0].set_ylabel(params["y_label"])

    # - axis labels
    ax[0,0].set_xlim(params["xlim_from"],params["xlim_to"])
    ax[0,0].set_ylim(params["ylim_from"],params["ylim_to"])

    # - general figure scaling
    fig.set_size_inches(8, 5)
    fig.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.30)

    fig.savefig(path + fname + '.png', dpi=1600)


def apply_shock_1(params, tx_shock, fv_shock):

    # - prepare some useful variables
    duration_shock_tx = params["n_tx_shock_end"] - params["n_tx_shock_start"]
    duration_shock_fv = params["n_fv_shock_end"] - params["n_fv_shock_start"]
    old_tx_level = params["onchain_vol_usd_baseline"]
    intermediate_tx_level = old_tx_level + tx_shock
    new_tx_level = intermediate_tx_level + tx_shock
    old_fv_level = params["fv_iter_start"]
    new_fv_level = old_fv_level + fv_shock * duration_shock_fv

    # - create example
    simulation_input = create_example_data(params)

    # - shock fv
    simulation_input["fv"][params["n_fv_shock_start"]:params["n_fv_shock_end"]] = [old_fv_level + fv_shock * i
    for i in
        range(0, duration_shock_fv)]
    simulation_input["fv"][params["n_fv_shock_end"]:] = [new_fv_level] * (
    len(simulation_input["fv"][params["n_fv_shock_end"]:]))

    # - shock vol
    simulation_input["onchain_vol_usd"][params["n_tx_shock_start"]:params["n_tx_shock_end"]] = [intermediate_tx_level] * duration_shock_tx

    return simulation_input


def gather_simulation_components(simulation_input, simulation_output):
    basic_variables = {
        't': [i for i in range(1, len(simulation_input) + 1)]
        , 'm_total': simulation_input.m_total
        , 'cash_balance': simulation_input.onchain_vol_usd / simulation_input.v_circ
        , 'fv': simulation_input.fv
    }
    basic_variables = pd.DataFrame(basic_variables)
    basic_variables_ordered = basic_variables[['t', 'm_total', 'cash_balance', 'fv']]
    plotting_data = pd.concat([basic_variables_ordered
                                  , simulation_output["components"]
                                  , simulation_output["prices"]], axis=1, sort=False)
    return(plotting_data)


def make_simulation_shock_both(path, fname, params):

    data = list()
    plot_cols = list()
    for fv_shock in params["fv_shock"]:
        for tx_shock in params["tx_shock"]:

            # -- create shocked dataframe
            simulation_input = apply_shock_1(params, tx_shock, fv_shock)
            simulation_output = simulation(params=params, simulation_input=simulation_input)

            # -- Prepare simulation results for plotting
            plotting_data = gather_simulation_components(simulation_input, simulation_output)
            data.append(plotting_data)

            # -- Gather inputs for plotting
            # - data for subplots
            basic_variables = []
            plot_cols_of_interest = strip_column_names(plotting_data.columns, variable_id="hodl",
                                                              additional_restrictions=[])
            plot_cols_of_interest = basic_variables + plot_cols_of_interest
            plot_cols.append(plot_cols_of_interest)


    # - styling
    line_styles = ["solid", "dashdot", "dashed", "dotted", "dashdot", "dashed", "dotted",]
    line_widths = [1, 1, 1, 1, 1, 1, 1, 1]
    colors = ["black", "dimgrey", "dimgrey", "dimgrey","dimgrey", "dimgrey", "dimgrey"]

    # - assemble plot
    fig, axs = plt.subplots(1, 2)
    # - plot price simulations
    for p in range(len(plot_cols[0])):
        axs[0].plot('t'
                       , plot_cols[0][p]
                       , data=data[0]#[(params["n_fv_shock_end"]+1):]
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label=get_labels_from_variables(plot_cols[0][p], typ="qm", typ_label=r"$q_m$"
                                                         )
                    )
    for p in range(len(plot_cols[1])):
        axs[1].plot('t'
                       , plot_cols[1][p]
                       , data=data[1]#[(params["n_fv_shock_end"]+1):]
                       , marker=None
                       , color=colors[p]
                       , linewidth=line_widths[p]
                       , linestyle=line_styles[p]
                       , label=get_labels_from_variables(plot_cols[1][p], typ="qm", typ_label=r"$q_m$"
                                                         )
                    )
    # - title
    axs[0].set_title(r"Panel (1): $\Delta B_{50,51} = - \Delta B_{51,52} = 100 USD$" + "\n" + "at $Z_{50} = 550$ $CC$", fontsize=10)
    axs[1].set_title(r"Panel (2): $\Delta B_{50,51} = - \Delta B_{51,52} = 100 USD$" + "\n" + "at $Z_{50} = 750$ $CC$", fontsize=10)

    # - legend
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                     fancybox=False, shadow=False, ncol=2, frameon=False)
    axs[0].set_ylim(params["ylim_from"], params["ylim_to"])
    axs[1].set_ylim(params["ylim_from"], params["ylim_to"])
    axs[0].set_ylabel(r'$Z_t$ in $CC$')
    axs[1].set_ylabel(r'$Z_t$ in $CC$')


    # - general figure scaling
    fig.set_size_inches(params["width"], params["height"])
    fig.subplots_adjust(wspace=0.5, hspace=0.5, bottom=params["fig_space_bottom"])

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def instability_simulations(params):

    i_append = 0
    results = pd.DataFrame(columns=("fv_lin_increase"
                                           , "tx_shock"
                                           , "qm"
                                           , "qv"
                                           , "cm"
                                           , "cv"
                                           , "hodl_ratio"
                                           , "price_var_before_shock"
                                           , "price_var_after_shock"
                                           , "hodl_var_before_shock"
                                           , "hodl_var_after_shock"
                                           , "price_shockperiod_var"
                                           , "hodl_shockperiod_var"))
    for fv_shock in params["fv_shock"]:
        for tx_shock in params["tx_shock"]:

            # -- create shocked dataframe
            simulation_input = apply_shock_1(params, tx_shock, fv_shock)
            simulation_output = simulation(params=params, simulation_input=simulation_input)

            for c in range(len(simulation_output["prices"].columns)):

                # - check if indexing works well
                if re.sub("prices", '', simulation_output["prices"].columns[c]) != \
                        re.sub("hodl", '', simulation_output["components"].columns[c]):
                    raise Exception("Simulation columns mixed up! Results will be wrong.")

                # - preparations
                shock_period_start = params["n_tx_shock_start"] - 1
                shock_period_end = params["n"] - params["staging_tail"] - 1
                staging_head_start = shock_period_start - params["staging_head"]
                staging_head_end = shock_period_start - 1
                staging_tail_start = params["n"] - params["staging_tail"]
                staging_tail_end = params["n"]

                # - calc values
                hodl_var_before_shock = (simulation_output["components"].iloc[staging_head_start:staging_head_end, [c]]).var()
                price_var_before_shock = (simulation_output["prices"].iloc[staging_head_start:staging_head_end, [c]]).var()

                hodl_var_after_shock = (simulation_output["components"].iloc[staging_tail_start:staging_tail_end, [c]]).var()
                price_var_after_shock = (simulation_output["prices"].iloc[staging_tail_start:staging_tail_end, [c]]).var()

                price_shockperiod_var = (simulation_output["prices"].iloc[shock_period_start:shock_period_end, [c]]).var()
                hodl_shockperiod_var = (simulation_output["prices"].iloc[shock_period_start:shock_period_end, [c]]).var()

                hodl_ratio = (simulation_output["components"].iloc[staging_head_start:staging_head_end, [c]]/params["m_total_baseline"]).mean()*100

                # - gather results
                results.loc[i_append] = [fv_shock
                    , tx_shock
                    , float(strip_packed_parameters(simulation_output["prices"].columns[c])["qm"])
                    , float(strip_packed_parameters(simulation_output["prices"].columns[c])["qv"])
                    , float(strip_packed_parameters(simulation_output["prices"].columns[c])["cm"])
                    , float(strip_packed_parameters(simulation_output["prices"].columns[c])["cv"])
                    , float(hodl_ratio)
                    , float(price_var_before_shock)
                    , float(price_var_after_shock)
                    , float(hodl_var_before_shock)
                    , float(hodl_var_after_shock)
                    , float(price_shockperiod_var)
                    , float(hodl_shockperiod_var)]

                # - increment iterable
                i_append += 1
    return results

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def make_instability_simulations_plot(path, fname, params):

    df_raw = instability_simulations(params=params)

    # - styling
    marker_styles = ["o", "v", "s", "D", "X", ">", "<", "H", "d", "p", "^", "."]
    marker_colors = ["black", "dimgray", "darkgray", "lightgray", "gainsboro"]
    line_styles = ["solid", "dashed", "dashdot", "solid", "dashed" , "solid", "dotted","solid", "dashed" , "solid", "dotted"]
    line_widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    line_colors = ["black", "dimgray", "darkgray", "lightgray", "gainsboro"]

    # - assemble plot
    for i, qm in enumerate(params["amplitude_param_m"]):
        plotting_data = df_raw[df_raw["qm"] == qm].reset_index(drop=True)
        plt.plot("hodl_ratio"
                 , "price_shockperiod_var"
                 , data=plotting_data
                 , marker="."
                 , markersize=7
                 , color=line_colors[i]
                 , linestyle="dotted"
                 , label=r"$q_m = {}$".format(qm)
                 )
    # - axis titles
    plt.xlabel(r"$\frac{Z_{t}}{M_{t}}$ in %")
    plt.ylabel(r"$\sigma^{2}(S_{t})$")
    plt.ylim(params["ylim_from"],params["ylim_to"])
    plt.xlim(params["xlim_from"],params["xlim_to"])
    # - legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=10, frameon=False)

    # - general figure scaling
    fig = matplotlib.pyplot.gcf()
    fig.subplots_adjust(wspace=0.5, hspace=0.5, bottom=params["fig_space_bottom"])
    fig.set_size_inches(8, 5)
    # # -- save
    fig.savefig(path + fname + '.png', dpi=1600)
    plt.close(fig)


def make_simulation_story(path, fname, params):

    simulation_input = apply_shock_2(params)
    simulation_output = simulation_simple(params=params, simulation_input=simulation_input)
    fig = plotting_guts_mechanics(params=params, simulation_output=simulation_output, simulation_input=simulation_input)

    # -- save
    fig.savefig(path + fname + '.png', dpi=1600)


def apply_shock_2(params):

    # - prepare some useful variables
    duration_shock_tx      = params["n_tx_shock_end"] - params["n_tx_shock_start"]
    duration_shock_fv_up   = params["n_fv_shock_up_end"] - params["n_fv_shock_up_start"]
    duration_shock_fv_down = params["n_fv_shock_down_end"] - params["n_fv_shock_down_start"]

    old_tx_level = params["onchain_vol_usd_baseline"]
    new_tx_level = old_tx_level + params["tx_shock"] * duration_shock_tx
    old_fv_level = params["fv_iter_start"]
    intermediate_fv_level = old_fv_level + params["fv_shock_up"] * duration_shock_fv_up
    new_fv_level = intermediate_fv_level + params["fv_shock_down"] * duration_shock_fv_down

    # - create example
    simulation_input = create_example_data(params)

    # - shocks fv
    # -- up-shock
    simulation_input["fv"][params["n_fv_shock_up_start"]:params["n_fv_shock_up_end"]] = [old_fv_level + params["fv_shock_up"] * i for i in range(0, duration_shock_fv_up)]
    simulation_input["fv"][params["n_fv_shock_up_end"]:] = [intermediate_fv_level] * (len(simulation_input["fv"][params["n_fv_shock_up_end"]:]))
    # -- down-shock
    simulation_input["fv"][params["n_fv_shock_down_start"]:params["n_fv_shock_down_end"]] = [intermediate_fv_level + params["fv_shock_down"] * i for i in range(0, duration_shock_fv_down)]
    simulation_input["fv"][params["n_fv_shock_down_end"]:] = [new_fv_level] * (len(simulation_input["fv"][params["n_fv_shock_down_end"]:]))

    # - shock vol
    simulation_input["onchain_vol_usd"][params["n_tx_shock_start"]:params["n_tx_shock_end"]] = [old_tx_level + params["tx_shock"] * i for i in range(0, duration_shock_tx)]
    simulation_input["onchain_vol_usd"][params["n_tx_shock_end"]:] = [new_tx_level] * (len(simulation_input["onchain_vol_usd"][params["n_tx_shock_end"]:]))

    return simulation_input

