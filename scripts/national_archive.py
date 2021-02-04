"""
GADI driver script for the Quality control check of dual-polarization variables.

@title: radar_qcchecks
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology
@creation: 02/02/2021
@date: 04/02/2021

.. autosummary::
    :toctree: generated/

    check_rid
    extract_zip
    get_radar_archive_file
    remove
    buffer
    process_quality_control
    main
"""
import gc
import os
import sys
import time
import zipfile
import argparse
import datetime
import warnings
import traceback
from typing import Dict

import crayons
import numpy as np
import pandas as pd
import dask
import dask.bag as db

import radar_qcchecks


def check_rid() -> bool:
    """
    Check if the Radar ID provided exists.
    """
    indir = f"/g/data/rq0/level_1/odim_pvol/{RID:02}"
    return os.path.exists(indir)


def extract_zip(inzip: str, path: str = "/scratch/kl02/vhl548/unzipdir"):
    """
    Extract content of a zipfile inside a given directory.

    Parameters:
    ===========
    inzip: str
        Input zip file.
    path: str
        Output path.

    Returns:
    ========
    namelist: List
        List of files extracted from  the zip.
    """
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return namelist


def get_radar_archive_file(date, rid: int) -> str:
    """
    Return the archive containing the radar file for a given radar ID and a
    given date.
    Parameters:
    ===========
    date: datetime
        Date.
    Returns:
    ========
    file: str
        Radar archive if it exists at the given date.
    """
    try:
        date = pd.Timestamp(date)
    except Exception:
        traceback.print_exc()
        pass
    datestr = date.strftime("%Y%m%d")
    file = f"/g/data/rq0/level_1/odim_pvol/{rid:02}/{date.year}/vol/{rid:02}_{datestr}.pvol.zip"
    if not os.path.exists(file):
        return None

    return file


def remove(flist: list) -> None:
    """
    Remove file if it exists.
    """
    flist = [f for f in flist if f is not None]
    for f in flist:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def buffer(infile: str) -> Dict:
    """
    Buffer function to catch and kill errors about missing Sun hit.

    Parameters:
    ===========
    infile: str
        Input radar file.

    Returns:
    ========
    rslt: Dict
        Quality checks results from qccheck_radar_odim.
    """
    try:
        rslt = radar_qcchecks.qccheck_radar_odim(infile)
    except Exception:
        print(f"Problem with file {infile}.")
        traceback.print_exc()
        return None

    return rslt


def process_quality_control(rid: int, date: pd.Timestamp, outpath: str) -> None:
    """
    Driver for processing the quality control. Find the data for given radar ID
    and dates on GADI. Unzip it and send it to the radar_qcchecks function.
    Also takes care of saving the output results into a daily csv file.

    Parameters:
    ===========
    rid: int
        Radar rapic ID.
    date: pd.Timestamp
        Date to process.
    outpath: str
        Output path directory.
    """
    datestr = date.strftime("%Y%m%d")
    fname = f"{rid}_stats_{datestr}.csv"
    fname = os.path.join(outpath, fname)
    if os.path.isfile(fname):
        print("Output file already exists. Doing nothing.")
        return None

    inzip = get_radar_archive_file(date, rid)
    if inzip is None:
        print(f"Couldn't get zip archive for {date} and radar {rid}.")
        return None

    flist = extract_zip(inzip)
    print(f"Found {len(flist)} files for radar {rid} on the {datestr}.")
    try:
        bag = db.from_sequence(flist).map(buffer)
        rslt = bag.compute()

        rslt = [r for r in rslt if r is not None]
        if len(rslt) == 0:
            raise ValueError("No rain today.")

        df = pd.DataFrame(rslt[0], index=[0])
        for r in rslt[1:]:
            df = df.append(r, ignore_index=True)

        df.set_index("time").to_csv(fname, float_format="%g")
        print(f"{fname} saved.")
    except Exception:
        traceback.print_exc()
    finally:
        remove(flist)

    gc.collect()
    return None


def main(start_date: datetime.datetime, end_date: datetime.datetime) -> None:
    tick = time.time()
    date_range = pd.date_range(start_date, end_date)
    for date in date_range:
        process_quality_control(RID, date, OUTPATH)
    tock = time.time()
    print(crayons.magenta(f"Process finished in {tock - tick:0.4}s."))

    return None


if __name__ == "__main__":
    parser_description = "Solar calibration of radar in the National radar archive."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-r", "--rid", dest="rid", type=int, required=True, help="Radar ID number.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="/scratch/kl02/vhl548/qcchecks/",
        type=str,
        help="Output directory for csv files.",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        dest="start_date",
        default=None,
        type=str,
        help="Start date for radar processing.",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--end-date",
        dest="end_date",
        default=None,
        type=str,
        help="End date for radar processing.",
        required=True,
    )

    args = parser.parse_args()
    RID = args.rid
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTPATH = args.output
    ZIPDIR = "/scratch/kl02/vhl548/unzipdir/"

    if not check_rid():
        parser.error("Invalid Radar ID.")
        sys.exit()

    try:
        start = datetime.datetime.strptime(START_DATE, "%Y%m%d")
        end = datetime.datetime.strptime(END_DATE, "%Y%m%d")
        if start > end:
            parser.error("End date older than start date.")
    except ValueError:
        parser.error("Invalid dates.")
        sys.exit()

    print(crayons.green(f"Processing QC checks for radar {RID}."))
    print(crayons.green(f"Between {START_DATE} and {END_DATE}."))
    print(crayons.green(f"Data will be saved in {OUTPATH}."))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(start, end)
