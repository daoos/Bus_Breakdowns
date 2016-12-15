import pandas as pd
import numpy as np
import datetime
import pickle
import multiprocessing
from tqdm import tqdm
import sys
import time
import os

# ##GLOBALS## #
n = None  # number of rows to read in
time_horizon = [12]  # window (in hours) before the breakdown. Make sure left is less than right
minimum_time = 1  # duration for entries without a duration in seconds
use_duration = True  # switch to amplifying faults by the duration of the fault (may undervalue low duration errors)
use_breakdownTypes = False  # whether to create a single model for breakdowns or for each type of breakdowns (Eng, etc)
use_errorSystem = True  # use the general location or the specific error
list_of_valid_breakdowns = ["B11-COLD BUS", "B11-COLD BUS", "B11-HOT BUS", "B11-NO A/C", "B11-NO DEFROSTER",
                            "B11-OTHER", "B12-LOW AIR LIGHT", "B12-PRESSURE HIGH", "B12-PRESSURE LOW",
                            "B13-BRAKES GRAB", "B13-OTHER", "B13-PARK BRAKE DEFEC", "B13-PEDAL STICK DOWN",
                            "B13-POPS UP", "B13-WEAK BRAKES", "B13-WILL NOT RELEASE", "B16-LEANS", "B16-LOW AIR RIDE",
                            "B16-LOW SUSPENSION", "B16-NOISY SHOCKS", "B16-OTHER", "B17-JUMP GEAR",
                            "B17-JUMP GEAR MILD", "B17-JUMP GEAR SEVERE", "B17-LIGHT ON", "B17-LOW TORQUE",
                            "B17-NO DIRECT DRIVE", "B17-NOT GO IN GEAR", "B17-OTHER", "B17-SLOW BUS",
                            "B17-WILL NOT PULL", "B17-WILL NOT SHIFT", "B18-LOW COOLANT LIGH", "B18-OVER HEAT",
                            "B19-BUZZER", "B19-EMERGENCY HANDLE", "B19-FRONT DOOR", "B19-OTHER", "B19-R DR ALARM", "B19-R DR INTERLOCK", "B19-R DR OTHER",
                            "B19-R DR WONT CLOSE", "B19-R DR WONT OPEN", "B20-DEAD BATTERIES", "B20-GENERATOR LIGHT",
                            "B20-NO POWER", "B20-UNKNOWN BUZZER", "B21-CHK ENGINE LIGHT", "B21-DEAD MOTOR",
                            "B21-EXCESSIVE SMOKE", "B21-LOUD NOISE", "B21-MOTOR DIES", "B21-OTHER", "B21-SMOKE IN REAR",
                            "B21-WONT START", "B22-ACCELER PEDAL", "B22-FUEL LEAK", "B22-FUMES", "B22-NO FUEL",
                            "B23-HEADLIGHT DEFECT", "B23-INTERIOR LIGHTS", "B23-MARKER LIGHTS", "B23-OTHER",
                            "B23-TAILLIGHTS", "B23-TURN SIGNALS", "B26-POWER", "B26-PULLS", "B27-FLAT FRONT WHEEL",
                            "B27-FLAT FRONT WHEEL", "B27-FLAT R WHL INSID", "B27-FLAT R WHL INSID",
                            "B27-FLAT R WHL OUTSI", "B27-FLAT R WHL OUTSI", "B27-LOOSE LUGS", "B27-LOOSE LUGS",
                            "B27-LOOSE WHEELS", "B27-LOOSE WHEELS", "B27-OTHER", "B27-OTHER", "B27-SMOKING",
                            "B27-SMOKING", "B31-ALARM NOT WORKIN", "B31-FALSE SIGNALS", "B31-OTHER", "B40-NO POWER",
                            "B40-OTHER", "B40-PLATFORM WONT ST", "B40-PLATFRM WONT DEP", "B40-WONT MOVE"]


def a(x):
    print(x.iloc[:4, :])
    exit()


def isFloat(x):  # tests if something can be converted to a float
    try:
        float(x)
        return True
    except ValueError:
        return False


def get_dataframe(name, file_path, nrows, cols, colnames, date_cols, csv_type, delimiter=","): #cache the Breakdowns
    if os.path.exists(name + '.pkl'):
        print(name + " csv was cached")
        return pickle.load(open(name + '.pkl', 'rb'))
    else:
        print(name+ ' is being cached. This may take a few minutes.')
        dataframe = pd.read_csv(file_path,
                                nrows=nrows,
                                usecols=cols,
                                names=colnames,
                                parse_dates=date_cols,
                                error_bad_lines=False,
                                low_memory=True,
                                delimiter=delimiter)
        dataframe = dataframe.drop_duplicates()
        if csv_type == "Breakdown":
            dataframe = dataframe[[x in list_of_valid_breakdowns for x in dataframe['breakdown']]]
        else:
            Faults['duration'] = [float(x) if isFloat(x) else minimum_time for x in Faults['duration']]
        pickle.dump(dataframe, open(name + '.pkl', 'wb'))
        print(name + " caching Finished")
        return dataframe


def generateDataframes(use_errorSystem, n): #opens the documents necessary for the data manipulation
    errorIndex = {True: [7, 15], False: [8, 16]}[use_errorSystem]
    system_break = 6
    Breakdowns = get_dataframe(name="Breakdowns",
                               file_path='MH_PM_CAD_AVM_SRWO_MASH_2016.csv',
                               nrows=None,
                               cols=[1, 3, 4, system_break],
                               colnames=['BreakdownType', 'bus_id', 'date', 'breakdown'],
                               date_cols=[2],
                               csv_type="Breakdown")
    test_faults = get_dataframe(name="test_faults",
                            file_path='MH_PM_AVM_RAW_2016.csv',
                            nrows=n,
                            cols=[0, 2, errorIndex[0], 10],
                            colnames=['bus_id', 'date', 'fault', 'duration'],
                            date_cols=[1],
                            csv_type="Faults")
    predictor_faults = get_dataframe(name="test_faults",
                            file_path='MH_PM_AVM_RAW_OCT2016.csv',
                            nrows=n,
                            cols=[0, 2, errorIndex[0], 10],
                            colnames=['bus_id', 'date', 'fault', 'duration'],
                            date_cols=[1],
                            csv_type="Faults",
                            delimiter="|")
    Faults = pd.concat([test_faults, predictor_faults])
    return Faults.reset_index(drop=True), Breakdowns.reset_index(drop=True)


def create_snapshots(df, id, Breakdowns):
    bus_breakdates = Breakdowns[[str(x) == str(id) for x in Breakdowns['bus_id']]]['date']
    all_snapshot_df = pd.DataFrame(columns=['broke_down'] + list(df.columns))
    for date in df['date'].drop_duplicates():
        snapshot_df = df[[date - datetime.timedelta(hours=(time_horizon[0])) <= x <= date for x in df['date']]].copy()
        snapshot_df = snapshot_df.sum()  # converts id and date to string :(
        snapshot_df['bus_id'] = id
        snapshot_df['date'] = str(date)  # can't go directly to replace with date. Idk why, but everything works out.
        snapshot_df['date'] = date
        if len([x for x in bus_breakdates if date + datetime.timedelta(hours=time_horizon[0]) >= x >= date]) > 0:
                snapshot_df['broke_down'] = 1
        else:
                snapshot_df['broke_down'] = 0
        all_snapshot_df = all_snapshot_df.append(snapshot_df, ignore_index=True)
    with open('output/result_data' + str(id) + '.pkl', 'wb') as f:
        pickle.dump(all_snapshot_df, f)
        f.close()


if __name__ == "__main__":
    print('Program beginning')
    Faults, Breakdowns = generateDataframes(use_errorSystem, n)
    print('frames loaded')
    Faults = pd.get_dummies(Faults, columns=['fault'])
    if use_duration:
        Faults.loc[:, 3:] = Faults.iloc[:, 3:].rmul(Faults['duration'], axis='index').astype(np.float32)
    Faults.drop('duration', axis=1, inplace=True)
    faults_by_bus = Faults.groupby('bus_id')
    print('grouped by faults')
    pickle.dump(faults_by_bus, open('thing.pkl', 'wb')),
    pool = multiprocessing.Pool()
    for bus_id, bus_df in tqdm(faults_by_bus):
            pool.apply(create_snapshots, args=(bus_df.copy(), bus_id, Breakdowns.copy(),))
