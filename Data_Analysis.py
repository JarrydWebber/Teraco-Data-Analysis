import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import itertools
import configparser
import pathlib
import math

config = configparser.ConfigParser()
config.read('config.ini')
datadir = pathlib.Path(config['paths']['datadir']).expanduser()
basedir = pathlib.Path(config['paths']['basedir']).expanduser()
savedir = pathlib.Path(config['paths']['savedir']).expanduser()

baseline_start = config['baseline']['start']
baseline_end = config['baseline']['end']
analysis_start = config['analysis']['start']
analysis_end = config['analysis']['end']

target_r2 = float(config['regression']['target_r2'])
overfitting = config['regression']['overfitting']
config_periods = config['regression']['periods'].replace(" ", "").split(',')

if overfitting == "True":
    overfitting = True
else:
    overfitting = False

testing_graphs_print = config['testing']['extra_graphs']

if testing_graphs_print == "True":
    testing_graphs_print = True
else:
    testing_graphs_print = False

### DEFS

def save_figure(save_name, title=None, x_axis=None, y_axis=None):
    if title != None:
        plt.title(title)

    if x_axis != None:
        plt.xlabel(x_axis)

    if y_axis != None:
        plt.ylabel(y_axis)

    if x_axis == "Date":
        plt.tick_params(axis = 'x', rotation=90)

    path = savedir
    local_save_path = os.path.join(path, save_name + ".jpg")
    plt.savefig(local_save_path, bbox_inches='tight', dpi=200)


def regression(Y_data, X_data, Timeframe, Aggregation_Period, Aggregation_method="mean"):
    start, end = Timeframe
    # print (start, end)
    X_data = X_data[start:end]
    Y_data = Y_data[start:end]

    if Aggregation_method == "mean":
        x_reg = X_data.resample(Aggregation_Period).mean()
        y_reg = Y_data.resample(Aggregation_Period).mean()

    else:
        x_reg = X_data.resample(Aggregation_Period).sum()
        y_reg = Y_data.resample(Aggregation_Period).sum()

    # x_reg.interpolate()

    inf = np.isinf(x_reg).sum().sum()
    nans = x_reg.isna().sum().sum()
    if inf + nans > 0:
        # print(inf, nans)
        x_reg.interpolate(method="backfill", inplace=True)
    x_reg = sm.add_constant(x_reg)

    results = sm.OLS(y_reg, x_reg).fit()

    return (results)


def parameterisation(Y_data, X_data, Timeframe, Aggregation_Period, Aggregation_method="mean", target_R=0.1, overfitting=True):
    params = []
    total_columns = len(X_data.columns)
    current = 1
    for column in X_data.columns:
        # print("Regression over column {0} of {1}. {2}.".format(current, total_columns, column))
        current += 1
        result = regression(Y_data, X_data[column], Timeframe, Aggregation_Period, Aggregation_method)
        # print (column, result.rsquared)
        if result.rsquared >= target_R:
            params.append(column)

    perms_list = []
    if overfitting:
        max_params = max(1, math.floor(len(Y_data[date1:date2].resample(Aggregation_Period).mean())/15))
        print (max_params)
    else:
        max_params = 4
    print("PARAMS", max_params, len(Y_data))
    for length in range(max_params):
        length += 1
        perms = itertools.combinations(params, length)
        for i in perms:
            perms_list.append(i)

    return (params, perms_list)


###
period_list = config_periods

date1 = analysis_start
date2 = analysis_end

baseline_date1 = baseline_start
baseline_date2 = baseline_end

savedir = os.path.join(savedir, "Outputs")
if not os.path.isdir(savedir):
    os.mkdir(savedir)

datadir = os.path.join(datadir, "Data")
directory = os.listdir(datadir)

# print (directory,)

if "Master.csv" not in directory:
    base_frame = pd.DataFrame()
    for file in directory:
        if file[-4:] == ".csv":
            temp_frame = pd.read_csv(os.path.join(datadir, file), delimiter=';', index_col="Timestamp", parse_dates=True)
            base_frame = pd.concat([base_frame, temp_frame])
        os.remove(os.path.join(datadir, file))

    base_frame = (base_frame.reset_index()
            .drop_duplicates(subset='Timestamp', keep='last')
            .set_index('Timestamp').sort_index())
    # base_frame.replace({",": "."}, regex=True, inplace=True)
    base_frame.to_csv(os.path.join(datadir, "Master.csv"))

elif len(os.listdir(datadir)) > 1:
    base_frame = pd.read_csv(os.path.join(datadir, "Master.csv"), index_col="Timestamp", parse_dates=True)
    for file in directory:
        if file[-4:] == ".csv" and file != "Master.csv":
            temp_frame = pd.read_csv(os.path.join(datadir, file), delimiter=';', index_col="Timestamp", parse_dates=True)
            base_frame = pd.concat([base_frame, temp_frame])
        os.remove(os.path.join(datadir, file))

    base_frame = (base_frame.reset_index()
            .drop_duplicates(subset='Timestamp', keep='last')
            .set_index('Timestamp').sort_index())
    # base_frame.replace({",": "."}, regex=True, inplace=True)
    base_frame.to_csv(os.path.join(datadir, "Master.csv"))

save_name = "Data Analysis Results.xlsx"
save_path = os.path.join(savedir, save_name)
writer = pd.ExcelWriter(save_path, engine='xlsxwriter')

name_list_location = os.path.join(basedir, "Name_list.csv")
nameing_list = pd.read_csv(name_list_location, index_col=0)

data_list_location = os.path.join(datadir, "Master.csv")



data = pd.read_csv(data_list_location, index_col="Timestamp", parse_dates=True)
data = (data.reset_index()
        .drop_duplicates(subset='Timestamp', keep='last')
        .set_index('Timestamp').sort_index())
data.replace({",": "."}, regex=True, inplace=True)

for col in data.columns:
    try:
        data[col] = pd.to_numeric(data[col])
    except:
        pass

    if data[col].mean() > 0:
        data[col].replace({0: np.nan})
        data[col].interpolate(inplace=True)

data = data.resample('h').max()

data.fillna(0, inplace=True)

data = data[1:-1]

baseline_data = data[baseline_date1:baseline_date2]

grouped_frame = pd.DataFrame(index=data.index)
diffed_data = pd.DataFrame(index=data.index)

TEMS_list = list(nameing_list["TEM"].unique())
TYPE_list = list(nameing_list["Type"].unique())
INCOME_list = list(nameing_list[nameing_list["Energy Flow"] == "Incoming"]["Type"].unique())
INCOME_list_total = list(["Total {0}".format(i) for i in INCOME_list])

incomers_list = list(nameing_list["Type"][nameing_list["Energy Flow"] == "Incoming"].unique())
outgoing_list = list(nameing_list["Type"][nameing_list["Energy Flow"] == "Outgoing"].unique())

print(data.columns)

for tem in TEMS_list:
    for t in TYPE_list:
        names = list(nameing_list["Full Name"][(nameing_list["Type"] == t) & (nameing_list["TEM"] == tem)].values)
        #        print ("{0} {1}".format(tem, t))
        # print (list(names))
        if any(["kWh" in name] for name in names):
            # print (data.columns)
            grouped_frame["{0} {1}".format(tem, t)] = data[names].sum(axis=1).diff()
            diffed_data[names] = data[names].diff().fillna(0)
#        print (data[names])

diffed_data["const"] = diffed_data[diffed_data.columns[0]] * 0 + 1

for t in TYPE_list:
    grouped_frame["{0} {1}".format("Total", t)] = grouped_frame[grouped_frame.filter(like=t).columns].sum(axis=1)
grouped_frame["{0} {1}".format("Total", "Incoming")] = grouped_frame[INCOME_list_total].sum(axis=1)
grouped_frame["{0} {1}".format("Total", "Small Power and Other")] = grouped_frame["Total Incoming"] - grouped_frame[
    "Total UPS"] - grouped_frame["Total HVAC"]

grouped_frame.fillna(0, inplace=True)

TEMS_list.append("Total")

###

annualised_data = grouped_frame.filter(like="Total").resample('y').sum()
x_locs = np.arange(1, (len(annualised_data.index.year)) + 1)
plt.plot(figsize=(10, 10))
plt.ticklabel_format(style='plain', useOffset=False)

plt.xticks(x_locs, annualised_data.index.year)
plt.bar(x_locs, annualised_data["Total Incoming"].values)
plt.gcf().set_size_inches(len(x_locs)*2, 10)
save_figure("Annual Power Consumption", "Annual Energy Usage", "Year", "Energy (kWh)")
plt.close()

annualised_data["Total Incoming"].to_excel(writer, sheet_name="Annualised Power Consumption")
print("Annual Power Consumption Graph Complete")

###

plt.plot(figsize=(10, 10))
plt.ticklabel_format(style='plain', useOffset=False)
pie_plot = []
labels = []
total_incoming = grouped_frame["{0} {1}".format("Total", "Incoming")][date1:date2].sum()
for incomer in incomers_list:
    incomer_sum = grouped_frame["Total {0}".format(incomer)][date1:date2].sum()
    pie_plot.append(incomer_sum)
    labels.append("{0} {1}%".format(incomer, round(incomer_sum / total_incoming, 3) * 100))
plt.pie(pie_plot, labels=labels)
save_figure("Supply Power Sources", "Energy Supply Breakdown")

print (pie_plot, labels)
pie_plot_frame = pd.DataFrame(data=[pie_plot], columns=INCOME_list)
pie_plot_frame.T.to_excel(writer, sheet_name="Incomer Sources")

plt.close()



print("Supply Power Sources Graph Complete")

###

plt.plot(figsize=(10, 10))
plt.ticklabel_format(style='plain', useOffset=False)
plt.xticks([1, 2, 3], ["HVAC", "IT Load", "Small Power and Other"])
bar_data = [(grouped_frame["Total HVAC"][date1:date2].sum() / grouped_frame["Total Incoming"][date1:date2].sum())*100,
            (grouped_frame["Total UPS"][date1:date2] .sum()/ grouped_frame["Total Incoming"][date1:date2].sum())*100,
            (grouped_frame["Total Small Power and Other"][date1:date2].sum() / grouped_frame["Total Incoming"][date1:date2].sum())*100]
plt.bar([1, 2, 3], bar_data)
save_figure("SEU Power Consumption Comparison", "Power Consumed by System", "System", "Power Consumed by System (%)")
plt.close()

bar_data_frame = pd.DataFrame(data=[bar_data], columns=["HVAC", "IT Load", "Small Power and Other"])
bar_data_frame.T.to_excel(writer, sheet_name="SEU Data")

print("SEU Power Consumption Comparison Graph Complete")

###
# Function Testing
# results = regression(grouped_frame["Total HVAC"], grouped_frame[["Total UPS"]], ["2021-01-01", "2022-12-31"], 'd')
# print(results.summary())

# important_params = parameterisation(grouped_frame["Total HVAC"], data.filter(regex="^((?!HVAC).)*$"), ["2021-01-01", "2022-12-31"], 'd', target_R=0.02)
# print(important_params[0])

# print(grouped_frame)

###

target_data_excel = pd.DataFrame()
predicted_data_excel = pd.DataFrame()
kept_target_data_excel = pd.DataFrame()
diffs_data_excel = pd.DataFrame()
cusum_data_excel = pd.DataFrame()
enpi_data_excel = pd.DataFrame()
pue_data_excel = pd.DataFrame()
target_pue_data_excel = pd.DataFrame()

regression_results = []

counter = 1

tems_used = []

for tem in TEMS_list:
    if "{0} HVAC".format(tem) in grouped_frame.columns:

        tems_used.append(tem)
        # try:

        # print("Tem {0} of {1}. {2}.". format(counter, len(TEMS_list), tem))
        counter += 1

        best_R = 0
        accuracy = 3
        target = grouped_frame["{0} HVAC".format(tem)]

        for period in period_list:

            counter2 = 1

            print("{0} HVAC".format(tem), period)
            important_params, perms_list = parameterisation(target,
                                                            diffed_data[nameing_list[(nameing_list["Type"] != "HVAC") & (
                                                                    nameing_list["Energy Flow"] == "Outgoing")][
                                                                "Full Name"]],
                                                            [baseline_date1, baseline_date2],
                                                            period,
                                                            target_R=target_r2, overfitting=overfitting)

            for pair in perms_list:

                print("\t Combination {0} of {1}. {2}.".format(counter2, len(perms_list), pair))
                counter2 += 1

                predictors = diffed_data[list(pair)]
                regression_result = regression(target, predictors, [baseline_date1, baseline_date2], period)
                # print(regression_result.rsquared)

                if round(regression_result.rsquared, accuracy) > best_R:
                    best_R = regression_result.rsquared
                    best_params = pair
                    best_period = period
                    best_results = regression_result

        print("regressions done")


        diffed_data["const"] = diffed_data["const"] * 0 + 1
        predicted_Y = best_results.predict(diffed_data[best_results.params.index].resample(best_period).mean())

        kept_target = target.resample(best_period).mean()[
            target.resample(best_period).mean() <= predicted_Y].copy().resample(best_period).mean()
        kept_predictors = diffed_data.resample(best_period).mean()[
            target.resample(best_period).mean() <= predicted_Y].copy().resample(best_period).mean()

        try:
            targeting_regression_result = regression(kept_target.fillna(0), kept_predictors[list(best_params)].fillna(0), [date1, date2],
                                                     best_period)

            regression_results.append([best_results, best_period, tem, targeting_regression_result])

            print (targeting_regression_result.summary())

            tem_frame_excel_target = pd.DataFrame(data=targeting_regression_result.params)
            tem_frame_excel_target.columns = ["R2 = " + str(targeting_regression_result.rsquared)]
            tem_frame_excel_target.to_excel(writer, sheet_name=tem + " Target Parameters " + best_period)

        except:
            pass

        # for r, period, tem in regression_results:
        #     print(tem, period, r.summary(), '\n')

        # print (targeting_regression_result.summary())

        plt.plot(figsize=(10, 10))
        plt.ticklabel_format(style='plain', useOffset=False)
        plt.plot(target.resample(best_period).mean()[date1:date2] , label="Actual")
        plt.plot(predicted_Y[date1:date2] , '.', label="Predicted")
        plt.plot(kept_target[date1:date2] , '.', label="Target")
        plt.legend()
        # plt.plot(targeting_regression_result.predict(kept_predictors[targeting_regression_result.params]))
        save_figure("targets {0}".format(tem), "targets {0} with agg period of \'{1}\'".format(tem, best_period), "Date",
                    "kWh")
        plt.close()

        temp_target = target.resample(best_period).mean().reset_index()
        temp_target.columns = ["Timestamp", "Actual HVAC Power kWh"]
        temp_target["TEM"] = tem

        temp_predicted_Y = predicted_Y.reset_index()
        temp_predicted_Y.columns = ["Timestamp", "Predicted HVAC Power kWh"]
        temp_predicted_Y["TEM"] = tem

        temp_kept_target = kept_target.reset_index()
        temp_kept_target.columns = ["Timestamp", "Target HVAC Power kWh"]
        temp_kept_target["TEM"] = tem

        target_data_excel = pd.concat([target_data_excel, temp_target])
        predicted_data_excel = pd.concat([predicted_data_excel, temp_predicted_Y])
        kept_target_data_excel = pd.concat([kept_target_data_excel, temp_kept_target])


        width = 1
        gap = 0.2
        number = 2
        target_trim = target[date1:date2]
        pred_trim = predicted_Y[date1:date2]
        kept_target_trim = kept_target[date1:date2]

        plt.plot(figsize=(10, 10))
        plt.ticklabel_format(style='plain', useOffset=False)
        x_ticks_1 = numpy.arange(gap / 2 + (width-gap)/number * 0, width * len(target_trim.resample(best_period).mean()) + gap/2, width)
        x_ticks_2 = numpy.arange(gap / 2 + (width-gap)/number * 1, width * len(pred_trim) + gap / 2, width)
        # x_ticks_3 = numpy.arange(gap / 2 + (width-gap)/number * 2, width * len(kept_target_trim) + gap / 2, width)
        # print (len(x_ticks_1), len(x_ticks_2), len(target_trim.resample(best_period).mean()))
        plt.bar(x_ticks_1, target_trim.resample(best_period).mean(), (width-gap)/number,align = "edge", label="Actual")
        plt.bar(x_ticks_2, pred_trim, (width-gap)/number,align = "edge", label="Predicted")
        # plt.bar(x_ticks_3, kept_target_trim, (width-gap)/number,align = "edge", label="Target")

        plt.xticks(numpy.arange(len(pred_trim))+width/2, pred_trim.index.date)

        plt.legend()
        # plt.plot(targeting_regression_result.predict(kept_predictors[targeting_regression_result.params]))
        save_figure("targets bar {0}".format(tem), "targets {0} with agg period of \'{1}\'".format(tem, best_period), "Date",
                    "kWh")
        plt.close()

        tem_frame_excel = pd.DataFrame(data=best_results.params)
        tem_frame_excel.columns = ["R2 = " + str(best_results.rsquared)]
        tem_frame_excel.to_excel(writer, sheet_name=tem + " Fitting Parameters " + best_period)

        # print(targeting_regression_result.summary())

        # print (target, predicted_Y)
        diff = target.resample(best_period).mean() - predicted_Y
        # print (diff)
        Cusum = diff[baseline_date1:].cumsum()

        temp_diff = diff.reset_index()
        temp_diff.columns = ["Timestamp", "Difference kWh"]
        temp_diff["TEM"] = tem

        temp_cusum = Cusum.reset_index()
        temp_cusum.columns = ["Timestamp", "Cusum kWh"]
        temp_cusum["TEM"] = tem

        diffs_data_excel = pd.concat([diffs_data_excel, temp_diff])
        cusum_data_excel = pd.concat([cusum_data_excel, temp_cusum])

        # print (diffed_data["{0} Cusum".format(tem)])

        plt.plot(figsize=(10, 10))
        plt.ticklabel_format(style='plain', useOffset=False)
        if len(Cusum[baseline_date1:]) < len(Cusum[date1:]):
            plt.plot(Cusum[date1:date2] - (Cusum[baseline_date1:][0]))
        else:
            # print(len(Cusum[baseline_date1:]), len(Cusum[date1:]))
            plt.plot(Cusum[date1:date2] - (Cusum[date1:][0]))
        save_figure("Cusum {0}".format(tem), "Cumulative Sum of {0} with agg period of \'{1}\'".format(tem, best_period), "Date",
                    "kWh")
        plt.close()

        # print ("test")

        enpi = target.resample(best_period).mean()/predicted_Y

        plt.plot(figsize=(10, 10))
        plt.ticklabel_format(style='plain', useOffset=False)
        plt.plot(enpi[date1:date2])
        save_figure("ENPI {0}".format(tem),
                    "ENPI {0} with agg period of \'{1}\'. Actual over Expected".format(tem, best_period), "Date",
                    "ENPI")
        plt.close()

        temp_enpi = enpi.reset_index()
        temp_enpi.columns = ["Timestamp", "ENPI"]
        temp_enpi["TEM"] = tem

        enpi_data_excel = pd.concat([enpi_data_excel, temp_enpi])

pue = grouped_frame["Total Incoming"].resample(best_period).mean() / grouped_frame["Total UPS"].resample(best_period).mean()
target_pue = (grouped_frame["Total UPS"].resample(best_period).mean() +
              grouped_frame["Total Small Power and Other"].resample(best_period).mean() + predicted_Y
              )/ grouped_frame["Total UPS"].resample(best_period).mean()

plt.plot(figsize=(10, 10))
plt.ticklabel_format(style='plain', useOffset=False)
plt.plot(pue[date1:date2], label="PUE")
plt.plot(target_pue[date1:date2], label="Target PUE")
plt.legend()
save_figure("PUE {0}".format(tem),
            "PUE {0} with agg period of \'{1}\'".format(tem, best_period), "Date",
            "PUE")
plt.close()

temp_pue = pue.reset_index()
temp_pue.columns = ["Timestamp", "PUE"]


temp_target_pue = target_pue.reset_index()
temp_target_pue.columns = ["Timestamp", "Target PUE"]

temp_pue["TEM"] = tem
temp_target_pue["TEM"] = tem

for tem in tems_used:
    f1 = temp_pue.copy()
    f2 = temp_target_pue.copy()

    f1["TEM"] = tem
    f2["TEM"] = tem

    pue_data_excel = pd.concat([pue_data_excel, f1])
    target_pue_data_excel = pd.concat([target_pue_data_excel, f2])



grouped_frames = [target_data_excel, predicted_data_excel, kept_target_data_excel,
                                 diffs_data_excel, cusum_data_excel, enpi_data_excel, pue_data_excel,
                                 target_pue_data_excel]

tems_frames = []
for tem in tems_used:
    tem_frame_grouped = pd.DataFrame()

    for frame in grouped_frames:
        new_frame = frame[frame["TEM"]==tem][frame.columns[0:2]]
        # print (new_frame)
        new_frame.set_index("Timestamp", inplace=True)
        tem_frame_grouped = pd.concat([tem_frame_grouped, new_frame], axis=1)
    tem_frame_grouped["TEM"]=tem
    tem_frame_grouped = tem_frame_grouped.reset_index()
    tems_frames.append(tem_frame_grouped)

overall_excel_frame = pd.concat(tems_frames)
column_ordering = ["Timestamp", "TEM"]
base_order = column_ordering.copy()
for column in overall_excel_frame.columns:

    if column not in base_order:
        column_ordering.append(column)
overall_excel_frame = overall_excel_frame[column_ordering]

overall_excel_frame.to_excel(writer, sheet_name="Plotting Data")

if testing_graphs_print:
    plt.plot(figsize=(10, 10))
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.plot(grouped_frame["Total UPS"][date1:date2], label=["Total UPS"])
    plt.plot(grouped_frame["Total HVAC"][date1:date2], label=["Total HVAC"])
    plt.plot(grouped_frame["Total Incoming"][date1:date2], label=["Total Incoming"])
    plt.legend()
    save_figure("Totals".format(tem),
                "Totalled Types", "Date",
                "kWh")
    plt.close()

    plt.plot(figsize=(10, 10))
    plt.ticklabel_format(style='plain', useOffset=False)
    for full, name in zip(nameing_list[nameing_list["Energy Flow"] == "Incoming"]["Full Name"],
                          nameing_list[nameing_list["Energy Flow"] == "Incoming"]["Name"]):
        plt.plot(diffed_data[full], label=[name])
    plt.legend()
    save_figure("Incomers".format(tem),
                "Incomers", "Date",
                "kWh")
    plt.close()

    plt.plot(figsize=(10, 10))
    plt.ticklabel_format(style='plain', useOffset=False)
    for full, name in zip(nameing_list[nameing_list["Type"] == "Generator"]["Full Name"],
                          nameing_list[nameing_list["Type"] == "Generator"]["Name"]):
        plt.plot(diffed_data[full], label=[name])
    plt.legend()
    save_figure("Gens".format(tem),
                "Gens", "Date",
                "kWh")
    plt.close()

    for full, name in zip(nameing_list[nameing_list["Type"] == "UPS"]["Full Name"],
                          nameing_list[nameing_list["Type"] == "UPS"]["Name"]):
        plt.plot(diffed_data[full], label=[name])
    plt.legend()
    save_figure("UPSs".format(tem),
                "UPSs", "Date",
                "kWh")
    plt.close()

    for full, name in zip(nameing_list[nameing_list["Type"] == "HVAC"]["Full Name"],
                          nameing_list[nameing_list["Type"] == "HVAC"]["Name"]):
        plt.plot(diffed_data[full], label=[name])
    plt.legend()
    save_figure("HVACs".format(tem),
                "HVACs", "Date",
                "kWh")
    plt.close()
    print(nameing_list)

writer.save()
print("\n \n \n \n \n DONE")

